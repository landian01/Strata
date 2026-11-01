# Direct llama-swap deployment on Windows

The runtime chain is llama-swap → Strata's native HTTP service → Strata's engine.
There is no additional Python owner or backend daemon. Strata's HTTP service is
itself Python; this integration does not replace it with another server.

## Scope

This is an opt-in deployment recipe for an existing, pinned two-model install.
It does not install a new engine, download or repack weights, or retune GPUs.
The original installed Terminal, engine, two model configurations and expert maps
are inputs, not edit targets. The staging tool checks their recorded hashes.

The foreground service is copied from the installed Terminal into a versioned
native-runtime snapshot. Only the reviewed native control patch is applied, so
fixes already present in the installed package are retained. Model configuration
copies have exactly the original bytes; private ports are command-line overrides.

## Why the small native patch exists

The original foreground service already drains admitted work and closes its
engine. On Windows without a console, however, llama-swap cannot request that path
through a normal Unix signal. Its default stop command is a forced `taskkill`, not
an orderly native shutdown.

The optional control feature in [the native service](../../serve/server.py) adds:

- A private, instance-bound file stop channel, including while the model loads.
- A one-shot stop client in [the control module](../../serve/control.py).
- A held Windows process identity and a fail-closed self Job before any engine is
  spawned. Parent loss requests shutdown; the deadline ends only that owned Job.
- An explicit result: a forced close or a startup abort is never reported as a
  successful, ready, graceful stop.
- Read-only model configuration: `/config` and `/settings` writes are refused,
  and saved shared settings are not loaded.

These features are off unless explicitly requested. The private control directory
must already have an owner-only protected ACL on a local ACL-capable volume. The
service rejects an unsafe directory; it does not silently change its permissions.
The staging tool creates this directory once. A sandbox that prohibits changing
ACLs needs explicit permission for that staging/test operation; do not weaken the
checks or treat skipped Windows tests as a pass.

The gateway still owns request routing, aliases, mutual exclusion, and model
process creation. Normal cross-model routing waits for the previous HTTP/SSE
request. Explicit Pause/unload is different: the native server's drain deadline
protects admitted requests, up to 300 seconds, not indefinitely.

## Stage, do not activate

Run with the existing embedded Python, from any working directory:

```powershell
D:\Strata\terminal\0.1.51\python\python.exe -B `
  D:\Strata\source\integrations\llama-swap\stage.py `
  --root D:\Strata --output D:\Strata\production\direct-api
```

The output directory must not exist. Staging copies only service/control code and
small deployment files, never the Python installation, engine/DLLs, model weights
or prepared packs. The source install is not modified. A partial or failed stage
is retained and must not be mistaken for an accepted deployment. The reviewed
[native overlay patch](native-control.patch) is versioned beside the staging tool,
so restaging does not depend on an uncommitted diff or an external evidence folder.
Gateway state and control credentials are stored under an owner-only runtime directory.

The deployment manifest records the fixed inputs, code assets and exact commands.
The two model commands start the native foreground service directly. Their
`cmdStop` starts the native one-shot stop client, not a custom backend wrapper.
The shared control file additionally prevents both model services from running at
once. Environment variables explicitly disable saved tuning and acquisition.
The one-shot entry point also clears inherited Strata tuning and CUDA renumbering.

## Verify before changing the entry point

```powershell
D:\Strata\terminal\0.1.51\python\python.exe -B `
  D:\Strata\source\integrations\llama-swap\verify.py `
  --deployment D:\Strata\production\direct-api --mode cpu `
  --output D:\Strata\production\validation\direct-cpu-NEW
```

CPU mode uses the real pinned gateway executable and native mock servers. It never
loads a model and chooses isolated loopback ports. Live mode uses the original two
models, their existing ports and unchanged engine settings; only run it after the
old service is stopped and enough RAM is free.

Checks include model aliases, an unknown model returning 404, same-model process
reuse, read-only settings, cross-model SSE completion, client disconnect recovery,
reverse switching, explicit Pause during a stream, abrupt gateway exit, released
ports, and protected file hashes. Retain failed runs as well as accepted ones.
These are lifecycle and deployment smoke checks, not broad quality or performance
benchmarks. Windows native control tests are in [the CPU-only unittest](../../serve/test_control.py).
The one-shot helper race/locking checks are in [test_entry.ps1](test_entry.ps1);
[verify_entry.py](verify_entry.py) tests real warm-up cancellation, Status during
loading, restart, a model answer, and the candidate's own Stop command.

## One-shot entry point

[Invoke-Strata.ps1](Invoke-Strata.ps1) starts the native gateway and exits. Status
only queries the gateway and never touches an unloaded model's upstream endpoint.
Warm intentionally loads the selected model through normal gateway routing. The
entry point works with built-in Windows PowerShell 5.1; PowerShell 7 is not needed.
Gateway Pause/Resume requests have an empty body, as their handlers expect. The
short launcher detaches with file-backed stdout/stderr; no logging pipe requires
a hidden owner process to remain alive.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\Strata\production\direct-api\Invoke-Strata.ps1 -Action Plan
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\Strata\production\direct-api\Invoke-Strata.ps1 -Action Start
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\Strata\production\direct-api\Invoke-Strata.ps1 -Action Status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\Strata\production\direct-api\Invoke-Strata.ps1 -Action Warm -Preset IQ4
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\Strata\production\direct-api\Invoke-Strata.ps1 -Action Stop
```

Stop first pauses the gateway and waits for native model shutdown. It verifies
that the backends and engine are gone before ending the identity-checked idle
gateway. Ending that empty gateway is not claimed to be a graceful model stop.
A process-identity or permission failure is reported, not bypassed with a PID-only
kill. The original foreground service and the original launchers remain available
for rollback.

## Activation and rollback

Only after CPU and live acceptance, back up and replace the root launch/stop entry
points with the staged candidates. Keep the previous scripts and their hashes in
a dated backup. Do not overwrite the old integration, original model JSON, expert
maps, engine or installed Terminal. Restore the backed-up entry points only after
the direct service is stopped; never run both entry points on the same ports.

A gateway upgrade or engine upgrade is a separate change with separate testing.
