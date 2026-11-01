# CPU-only tests for the one-shot entry's state/locking helpers. No gateway or model is started.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory,
      [string]$ValidationRoot = 'D:\Strata\production\validation')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$directory = [IO.Path]::GetFullPath($OutputDirectory)
$allowed = [IO.Path]::GetFullPath($ValidationRoot).TrimEnd('\') + '\'
if (-not $directory.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $directory)) {
    throw 'Tests require a NEW directory under ValidationRoot.'
}
$null = New-Item -ItemType Directory -Path $directory
$tokens=$null; $parseErrors=$null
$tree=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'Invoke-Strata.ps1'),[ref]$tokens,[ref]$parseErrors)
if ($parseErrors.Count) { throw 'Entry point contains syntax errors.' }
$functions = @{}
foreach ($node in $tree.FindAll({ param($ast) $ast -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    $functions[$node.Name] = $node.Extent.Text
    . ([scriptblock]::Create($node.Extent.Text))
}
function Test-Port {
    $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
    try { $listener.Start(); return $listener.LocalEndpoint.Port } finally { $listener.Stop() }
}
$nativePath=Join-Path $directory 'native.json'
$Manifest=[pscustomobject]@{ state_directory=$directory; control_file=$nativePath; port=(Test-Port);
    gateway='not-a-gateway'; bindings=@([pscustomobject]@{name='Fixture'; model='fixture'; aliases=@('iq3'); port=(Test-Port); command=@([Diagnostics.Process]::GetCurrentProcess().MainModule.FileName)},
                                      [pscustomobject]@{name='Other'; model='other'; aliases=@('iq4'); port=(Test-Port); command=@('unused')}) }
$BaseUrl='http://127.0.0.1:1'
$StatePath=Join-Path $directory 'gateway.json'
$LockPath=Join-Path $directory 'operation.lock'
$self=[Diagnostics.Process]::GetCurrentProcess()
$nativeRecord=@{kind='strata-native-control';pid=$self.Id;token=$self.StartTime.ToFileTimeUtc().ToString();nonce=('a'*64);ready=$false;state='loading'}
$recordJson=$nativeRecord | ConvertTo-Json -Compress
[IO.File]::WriteAllText($nativePath,$recordJson)
$results=[Collections.Generic.List[object]]::new()
function Passed([string]$Name) { $results.Add(@{name=$Name;status='passed'}); Write-Output ($Name+': passed') }
try {
    Save-State @{process_id=0;state='old'}
    $sharing=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $reader=[IO.File]::Open($StatePath,[IO.FileMode]::Open,[IO.FileAccess]::Read,$sharing)
    try { Save-State @{process_id=0;state='new'} } finally { $reader.Dispose() }
    if ((Read-State).state -ne 'new') { throw 'Atomic state replacement did not publish the new record.' }
    Passed 'atomic-save-with-concurrent-shared-reader'

    $script:nativeReads=0
    function Read-SharedText([string]$Path) {
        $script:nativeReads++
        if ($script:nativeReads -le 2) { return '{"pid":' }
        return $recordJson
    }
    $parsed=Read-NativeState
    if ($parsed.pid -ne $self.Id -or $script:nativeReads -ne 3) { throw 'Partial native JSON was not retried.' }
    Passed 'partial-native-json-is-retried'

    $script:nativeReads=0
    function Read-SharedText([string]$Path) { $script:nativeReads++; return '{"pid":' }
    $refused=$false
    try { $null=Read-NativeState } catch { $refused=$true }
    if (-not $refused -or $script:nativeReads -ne 20) { throw 'Persistent malformed native state was not refused.' }
    Passed 'persistent-malformed-state-is-not-stopped'

    $script:nativeReads=0
    function Read-SharedText([string]$Path) { $script:nativeReads++; throw [UnauthorizedAccessException]::new('fixture denied') }
    $refused=$false
    try { $null=Read-NativeState } catch [UnauthorizedAccessException] { $refused=$true }
    if (-not $refused -or $script:nativeReads -ne 1) { throw 'A permission error was retried or swallowed.' }
    Passed 'permission-errors-are-not-retried-or-hidden'
    . ([scriptblock]::Create($functions['Read-SharedText']))

    $native=Get-NativeProcess (Read-NativeState)
    try { if ($native.Id -ne $self.Id) { throw 'Held native identity did not match.' } } finally { $native.Dispose() }
    if ((Status).state -ne 'cleanup-pending') { throw 'Status falsely reported stopped with a live native service.' }
    Passed 'gateway-absent-live-native-status-is-pending'
    $refused=$false
    try { $null=Stop-Gateway } catch { $refused=$_.Exception.Message -like '*still cleaning up*' }
    if (-not $refused) { throw 'Stop claimed cleanup while the native service was alive.' }
    Passed 'gateway-absent-live-native-stop-is-refused'

    function Start-Gateway {
        $unexpected=$null
        try { $unexpected=[IO.File]::Open($LockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None) }
        catch [IO.IOException] { return @{ok=$true} }
        finally { if ($null -ne $unexpected) { $unexpected.Dispose() } }
        throw 'Start did not hold its operation lock.'
    }
    function Request-Gateway([string]$Path,[string]$Method='Get',[int]$Timeout=20) {
        $check=[IO.File]::Open($LockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        $check.Dispose()
        return @{ok=$true}
    }
    $warm=Warm-Model 'iq3'
    if (-not $warm.ok) { throw 'Warm did not return its result.' }
    Passed 'warm-http-wait-does-not-hold-operation-lock'

    $stopTemplate=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'Stop-Strata.cmd.in') -Raw
    if (-not $stopTemplate.Contains('@DEPLOYMENT@\Invoke-Strata.ps1') -or $stopTemplate.Contains('@ROOT@')) {
        throw 'Stop template is not bound to its own deployment.'
    }
    Passed 'candidate-stop-is-self-contained'
    @{ok=$true;tests=@($results);scope='CPU helper tests; no gateway, model or engine was started'} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $directory 'result.json') -Encoding utf8
} catch {
    @{ok=$false;tests=@($results);error=$_.Exception.Message} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $directory 'result.json') -Encoding utf8
    throw
} finally { $self.Dispose() }
