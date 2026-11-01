# One-shot local entry point. The only long-lived processes are llama-swap,
# Strata's native HTTP service, and its engine. No custom owner/backend daemon.
[CmdletBinding()]
param(
    [ValidateSet('Start', 'Stop', 'Status', 'Plan', 'Warm', 'List')]
    [string]$Action = 'Start',
    [string]$Preset
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ManifestPath = Join-Path $PSScriptRoot 'manifest.json'
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$BaseUrl = 'http://127.0.0.1:{0}' -f $Manifest.port
$StatePath = Join-Path $Manifest.state_directory 'gateway.json'
$LockPath = Join-Path $Manifest.state_directory 'operation.lock'

function Read-SharedText([string]$Path) {
    # Status readers must not prevent a concurrent atomic replacement on Windows.
    $sharing = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $sharing)
    try {
        $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
}

function Read-State {
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        return (Read-SharedText $StatePath) | ConvertFrom-Json
    }
    return $null
}

function Save-State($Value) {
    # This exact, non-secret state file is serialized by operation.lock.
    $expected = Join-Path $Manifest.state_directory 'gateway.json'
    if ([IO.Path]::GetFullPath($StatePath) -ne [IO.Path]::GetFullPath($expected)) {
        throw 'Unexpected gateway state path.'
    }
    $temporary = Join-Path $Manifest.state_directory ('gateway.{0}.tmp' -f [Guid]::NewGuid().ToString('N'))
    if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($temporary)) -ne [IO.Path]::GetFullPath($Manifest.state_directory)) {
        throw 'Unexpected temporary state path.'
    }
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        [IO.File]::Replace($temporary, $StatePath, [NullString]::Value)
    } else {
        [IO.File]::Move($temporary, $StatePath)
    }
}

function With-OperationLock([scriptblock]$Code) {
    $operation = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try { & $Code } finally { $operation.Dispose() }
}

function Read-NativeState {
    if (-not (Test-Path -LiteralPath $Manifest.control_file -PathType Leaf)) { return $null }
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        # Native status is published in place. Retry only a transient empty/partial JSON;
        # permissions, path errors and identity failures are never interpreted as 'stopped'.
        $raw = Read-SharedText $Manifest.control_file
        try {
            if ([string]::IsNullOrWhiteSpace($raw)) { throw [FormatException]::new('Native status is temporarily empty.') }
            return $raw | ConvertFrom-Json
        } catch [ArgumentException], [FormatException] {
            if ($attempt -eq 19) { throw }
            Start-Sleep -Milliseconds 25
        }
    }
}

function Get-NativeProcess($Record) {
    if ($null -eq $Record) { return $null }
    if ($Record.kind -ne 'strata-native-control' -or [int64]$Record.pid -le 0) { throw 'Invalid native control record.' }
    try { $nativeProcess = [Diagnostics.Process]::GetProcessById([int]$Record.pid) }
    catch [ArgumentException] { return $null }
    try {
        $null = $nativeProcess.Handle
        if ($nativeProcess.HasExited -or $nativeProcess.StartTime.ToFileTimeUtc().ToString() -ne [string]$Record.token) {
            $nativeProcess.Dispose()
            return $null
        }
        if (-not [string]::Equals($nativeProcess.MainModule.FileName, [string]$Manifest.bindings[0].command[0], [StringComparison]::OrdinalIgnoreCase)) {
            throw 'The control record does not identify the pinned native service.'
        }
        return $nativeProcess
    } catch {
        $nativeProcess.Dispose()
        throw
    }
}

function Assert-NoModelProcess([int]$WaitSeconds = 0) {
    $deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
    do {
        $native = Get-NativeProcess (Read-NativeState)
        $nativeAlive = $null -ne $native
        if ($nativeAlive) { $native.Dispose() }
        $engineAlive = @(Get-Process -Name strata -ErrorAction SilentlyContinue).Count -gt 0
        if (-not $nativeAlive -and -not $engineAlive) { return }
        if ([DateTime]::UtcNow -ge $deadline) {
            if ($nativeAlive) { throw 'The gateway is stopped, but its native model service is still cleaning up. Inspect Status; no cleanup success is claimed.' }
            throw 'An engine still exists; it is not taken over and cleanup is not claimed.'
        }
        # A signaled parent HANDLE can precede the last Job child releasing GPU resources.
        # Wait only for actual process liveness; identity/ACL/JSON errors above still fail immediately.
        Start-Sleep -Milliseconds 100
    } while ($true)
}

function Assert-Install {
    foreach ($item in @($Manifest.protected) + @($Manifest.assets)) {
        if (-not (Test-Path -LiteralPath $item.path -PathType Leaf)) { throw "Missing pinned file: $($item.path)" }
        $actual = (Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $item.sha256) { throw "Pinned file changed: $($item.path)" }
    }
}

function Assert-PortFree([int]$Port) {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    $listener.Server.ExclusiveAddressUse = $true
    try { $listener.Start() } finally { $listener.Stop() }
}

function Get-OwnedProcess($Record) {
    if ($null -eq $Record -or -not $Record.process_id) { return $null }
    try { $targetProcess = [Diagnostics.Process]::GetProcessById([int]$Record.process_id) }
    catch [ArgumentException] { return $null }
    try {
        # Open and retain a real Process handle before comparing creation time.
        $null = $targetProcess.Handle
        if ($targetProcess.HasExited) { $targetProcess.Dispose(); return $null }
        $started = $targetProcess.StartTime.ToUniversalTime().Ticks.ToString()
        $executable = $targetProcess.MainModule.FileName
        if ($started -ne [string]$Record.creation_ticks -or
            -not [string]::Equals($executable, [string]$Manifest.gateway, [StringComparison]::OrdinalIgnoreCase)) {
            $targetProcess.Dispose()
            return $null
        }
        return $targetProcess
    } catch {
        $targetProcess.Dispose()
        throw 'Cannot verify the owned gateway process; refusing to act on a bare PID.'
    }
}

function Request-Gateway([string]$Path, [string]$Method = 'Get', [int]$Timeout = 20) {
    # Use the built-in Windows PowerShell as well as PowerShell 7, always without a system proxy.
    $request = [Net.HttpWebRequest]::Create($BaseUrl + $Path)
    $request.Proxy = $null
    $request.Method = $Method.ToUpperInvariant()
    $request.Timeout = $Timeout * 1000
    $request.ReadWriteTimeout = $Timeout * 1000
    $request.KeepAlive = $false
    if ($Method -eq 'Post') {
        $request.ContentType = 'application/json'
        $request.ServicePoint.Expect100Continue = $false
        # These gateway control endpoints take no body. Sending an unread '{}' while
        # closing the connection can produce a Windows TCP reset after a successful pause.
        $request.ContentLength = 0
    }
    $response = $request.GetResponse()
    try {
        $reader = [IO.StreamReader]::new($response.GetResponseStream(), [Text.Encoding]::UTF8)
        try { return $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    } finally { $response.Dispose() }
}

function Assert-Gateway {
    $version = Request-Gateway '/api/version'
    $models = Request-Gateway '/v1/models'
    if ($version.version -ne $Manifest.gateway_version) { throw 'The gateway version does not match the pinned deployment.' }
    $expected = @($Manifest.bindings | ForEach-Object { $_.model } | Sort-Object)
    $actual = @($models.data | ForEach-Object { $_.id } | Sort-Object)
    if (($expected -join "`n") -ne ($actual -join "`n")) { throw 'The gateway model catalog does not match this deployment.' }
}

function Status {
    $record = Read-State
    $process = Get-OwnedProcess $record
    if ($null -eq $process) {
        $native = Get-NativeProcess (Read-NativeState)
        if ($null -ne $native) {
            try { return @{ ok = $true; mode = 'direct-native'; state = 'cleanup-pending'; native_process_id = $native.Id; api_url = $BaseUrl + '/v1' } }
            finally { $native.Dispose() }
        }
        return @{ ok = $true; mode = 'direct-native'; state = 'stopped'; api_url = $BaseUrl + '/v1' }
    }
    try {
        Assert-Gateway
        $running = Request-Gateway '/running'
        return @{ ok = $true; mode = 'direct-native'; state = 'ready'; process_id = $process.Id;
                  api_url = $BaseUrl + '/v1'; running = @($running.running); bindings = @($Manifest.bindings | ForEach-Object {
                      @{ name = $_.name; model = $_.model; aliases = $_.aliases; expected_slots = $_.expected_slots }
                  }) }
    } finally { $process.Dispose() }
}

function Start-Gateway {
    Assert-Install
    $record = Read-State
    $existing = Get-OwnedProcess $record
    if ($null -ne $existing) {
        try {
            Assert-Gateway
            $null = Request-Gateway '/api/models/resume' 'Post'
            return @{ ok = $true; reused = $true; mode = 'direct-native'; api_url = $BaseUrl + '/v1'; process_id = $existing.Id }
        } finally { $existing.Dispose() }
    }
    Assert-PortFree ([int]$Manifest.port)
    foreach ($binding in $Manifest.bindings) { Assert-PortFree ([int]$binding.port) }
    Assert-NoModelProcess
    # Never inherit a tuning shell's overrides or CUDA renumbering. Model env is explicit in the gateway config.
    foreach ($entry in Get-ChildItem Env: | Where-Object { $_.Name.StartsWith('STRATA_') -or $_.Name -in @('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTHONHOME', 'PYTHONPATH') }) {
        [Environment]::SetEnvironmentVariable($entry.Name, $null, 'Process')
    }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONNOUSERSITE = '1'
    $stdout = Join-Path $Manifest.directory 'logs/gateway.stdout.log'
    $stderr = Join-Path $Manifest.directory 'logs/gateway.stderr.log'
    # cmd's START detaches only the gateway and gives it real file handles for logs.
    # Start-Process -RedirectStandardOutput uses a parent-owned pipe, which cannot outlive this one-shot script.
    foreach ($path in @($Manifest.gateway, $Manifest.gateway_config, $stdout, $stderr)) {
        if ($path -match '[%!?^&|<>\r\n"]') { throw 'Deployment paths contain unsupported command-shell metacharacters.' }
    }
    $before = @((Get-Process -Name llama-swap -ErrorAction SilentlyContinue) | ForEach-Object { $_.Id })
    $launch = 'start "" /b "{0}" -config "{1}" -listen 127.0.0.1:{2} -watch-config=false 1>>"{3}" 2>>"{4}"' -f $Manifest.gateway, $Manifest.gateway_config, $Manifest.port, $stdout, $stderr
    # ShellExecute does not inherit this one-shot host's captured pipes. The short
    # CMD process exits after START; only the native gateway remains.
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $env:ComSpec
    $startInfo.Arguments = '/d /s /c "' + $launch + '"'
    $startInfo.WorkingDirectory = $Manifest.root
    $startInfo.UseShellExecute = $true
    $startInfo.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $starter = [Diagnostics.Process]::Start($startInfo)
    try {
        if (-not $starter.WaitForExit(10000)) { throw 'The one-shot gateway start command did not exit.' }
        if ($starter.ExitCode) { throw 'The one-shot gateway start command failed.' }
    } finally { $starter.Dispose() }
    $process = $null
    $findDeadline = [DateTime]::UtcNow.AddSeconds(10)
    while ($null -eq $process -and [DateTime]::UtcNow -lt $findDeadline) {
        $candidates = @((Get-Process -Name llama-swap -ErrorAction SilentlyContinue) | Where-Object {
            $_.Id -notin $before -and [string]::Equals($_.MainModule.FileName, [string]$Manifest.gateway, [StringComparison]::OrdinalIgnoreCase)
        })
        if ($candidates.Count -gt 1) { throw 'Multiple new gateways appeared; refusing to select one.' }
        if ($candidates.Count -eq 1) { $process = $candidates[0] } else { Start-Sleep -Milliseconds 100 }
    }
    if ($null -eq $process) { throw 'The gateway did not remain running; inspect its file logs.' }
    try {
        $null = $process.Handle
        $record = @{ schema_version = 1; mode = 'direct-native'; process_id = $process.Id;
                     creation_ticks = $process.StartTime.ToUniversalTime().Ticks.ToString(); executable = $Manifest.gateway;
                     state = 'starting'; started_at = [DateTimeOffset]::Now.ToString('o') }
        try { Save-State $record } catch {
            # This is the just-created, retained process identity, not an arbitrary PID.
            # Do not leave an unrecorded gateway running if its initial state cannot be published.
            if (-not $process.HasExited) { $process.Kill(); $null = $process.WaitForExit(10000) }
            throw
        }
        $deadline = [DateTime]::UtcNow.AddSeconds(45)
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($process.HasExited) { throw "Gateway exited with code $($process.ExitCode); see its logs." }
            try {
                Assert-Gateway
                $record.state = 'ready'
                Save-State $record
                return @{ ok = $true; reused = $false; mode = 'direct-native'; api_url = $BaseUrl + '/v1'; process_id = $process.Id }
            } catch {
                if ($process.HasExited) { throw }
                Start-Sleep -Milliseconds 150
            }
        }
        throw 'Gateway startup timed out; its identity record is retained. Inspect Status before retrying.'
    } finally { $process.Dispose() }
}

function Stop-Gateway {
    $record = Read-State
    $process = Get-OwnedProcess $record
    if ($null -eq $process) {
        Assert-PortFree ([int]$Manifest.port)
        foreach ($binding in $Manifest.bindings) { Assert-PortFree ([int]$binding.port) }
        Assert-NoModelProcess
        return @{ ok = $true; mode = 'direct-native'; state = 'stopped'; already_stopped = $true }
    }
    $native = $null
    $closeMode = 'no-loaded-model'
    try {
        Assert-Gateway
        $nativeRecord = Read-NativeState
        $native = Get-NativeProcess $nativeRecord
        # Pause rejects new inference and unloads through each model's native cmdStop.
        # Native stop drains admitted calls; taskkill is NOT the normal backend shutdown path.
        $null = Request-Gateway '/api/models/pause' 'Post' 450
        $running = Request-Gateway '/running'
        if (@($running.running).Count) { throw 'Backend unloading has not completed; the gateway is retained.' }
        foreach ($binding in $Manifest.bindings) { Assert-PortFree ([int]$binding.port) }
        Assert-NoModelProcess -WaitSeconds 10
        if ($null -ne $native) {
            if (-not $native.WaitForExit(1000)) { throw 'The native service has not exited; no cleanup success is claimed.' }
            $final = Read-NativeState
            if ($final.pid -ne $nativeRecord.pid -or $final.token -ne $nativeRecord.token -or $final.nonce -ne $nativeRecord.nonce) {
                throw 'The native instance changed during shutdown; a graceful stop is not confirmed.'
            }
            if ($final.state -eq 'stopped' -and $final.ready -eq $true -and $final.graceful -eq $true -and $final.exit_code -eq 0 -and $native.ExitCode -eq 0) {
                $closeMode = 'graceful'
            } elseif (-not $nativeRecord.ready -and -not $final.ready -and $final.state -eq 'aborted' -and $final.reason -eq 'nonready-abort') {
                $closeMode = 'loading-aborted'  # cancellation completed, NOT a ready model's graceful close
            } else {
                throw 'Native shutdown was forced or unconfirmed; the paused gateway is retained for inspection.'
            }
        }
        # All model processes have exited. End only the retained, identity-checked idle gateway.
        $process.Kill()
        if (-not $process.WaitForExit(10000)) { throw 'The idle gateway did not exit.' }
        Assert-PortFree ([int]$Manifest.port)
        $record.state = 'stopped'
        $record | Add-Member -NotePropertyName stopped_at -NotePropertyValue ([DateTimeOffset]::Now.ToString('o')) -Force
        Save-State $record
        return @{ ok = $true; mode = 'direct-native'; state = 'stopped'; models_unloaded = $true;
                  model_stop = $closeMode; graceful_model_stop = ($closeMode -eq 'graceful') }
    } finally {
        if ($null -ne $native) { $native.Dispose() }
        $process.Dispose()
    }
}

function Warm-Model([string]$Name) {
    if (-not $Name) { throw 'Warm needs -Preset IQ3 or IQ4.' }
    $binding = @($Manifest.bindings | Where-Object { $_.name -ieq $Name -or $_.model -ieq $Name -or @($_.aliases) -icontains $Name })
    if ($binding.Count -ne 1) { throw 'Unknown or ambiguous model name. Use List.' }
    $null = With-OperationLock { Start-Gateway }
    # The potentially long warm-up wait is outside the operation lock, so Status/Stop remain usable.
    $id = [Uri]::EscapeDataString($binding[0].model)
    $null = Request-Gateway ('/upstream/{0}/health' -f $id) 'Get' 1200
    return @{ ok = $true; mode = 'direct-native'; model = $binding[0].model; api_url = $BaseUrl + '/v1' }
}

try {
    $result = switch ($Action) {
        'Plan' { Assert-Install; @{ ok = $true; plan = $true; mode = 'direct-native'; api_url = $BaseUrl + '/v1'; bindings = $Manifest.bindings; models_unchanged = $true; engine_unchanged = [bool]$Manifest.models_and_engine_unchanged; models_and_engine_unchanged = [bool]$Manifest.models_and_engine_unchanged } }
        'List' { @{ ok = $true; bindings = $Manifest.bindings } }
        'Status' { Status }
        'Start' { With-OperationLock { Start-Gateway } }
        'Stop' { With-OperationLock { Stop-Gateway } }
        'Warm' { Warm-Model $Preset }
    }
    $result | ConvertTo-Json -Depth 12
} catch {
    @{ ok = $false; error = $_.Exception.Message; mode = 'direct-native' } | ConvertTo-Json -Depth 4
    exit 1
}
