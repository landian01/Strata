# Follow-up investigation of the 2% decode difference

2026-10-28. Full-model acceptance: **PASS**. No engine source, build flags,
model, or tuning settings were changed to obtain this result. The tested source
revision remains `2f8c82bed82a77deb2b5f51ca348848e4fb2ea24`.

The [earlier complete comparison](../2026-10-28-dual-gpu-source-parity/README.md)
remains part of the record: old 98.560 versus new 96.530 token/s, **-2.06%**, with
the merge withheld. It is exploratory evidence and is not silently discarded or
pooled into a different primary test after observing this follow-up.

## Attribution evidence

1. Free generation is not a fixed amount of model work. In the earlier comparison,
   new-build draft acceptance was 88.74% versus 89.57%, while primary cold-weight
   traffic was 30.388 versus 29.675 MB/output token. These are measured workload
   differences, not by themselves a causal attribution to code or random noise.
   For the same 40,960 output tokens per build, the original executed 8,853
   verification windows and the submitted build 8,992: **1.57% more windows**.
   Aggregate time per window was 46.943 versus 47.189 ms, **0.52% higher**.
   Their product exactly gives the 2.10% increase in time/token (2.06% lower
   throughput). This decomposes the observed workload/time difference; window
   shapes and cache states differ, so it does not isolate a causal code effect.
2. Merely freezing hot-expert residency did not reproduce the same free-generated
   reasoning even across repeated invocations of the original executable.
3. Existing CLI oracle/follow diagnostics fixed 8,151 prompt token IDs, 2,048
   continuation IDs and the T=1..8 window cycle. All five runs (three old, two new)
   produced the same **456 window residual hashes and argmax records**, SHA-256
   `4871b4276cf0e5a66a01f91a84ddfa1ab84c97549bd37f80e7cab0476bd3515e`.
   The first old run overlapped cuobjdump and was excluded from timing, with the
   exclusion recorded before its replacement. All numerical records were retained.
   The remaining two runs per build averaged 107.156 ms/window old and 108.930
   new: **-1.63%** relative throughput. This small diagnostic sample is negative
   evidence, not a pass. It uses static residency, no MTP and CLI execution, so
   its speed must not be presented as production throughput.
4. NVIDIA cuobjdump compared machine words, including scheduling fields, for 22
   profiled kernel variants per architecture (SM75 and SM120). **21/22 matched
   exactly on each architecture**, including selected expert math, weight fetch,
   wait, result publication and scatter kernels. Only `plan_gpu_tasks` differed.
   Its new static instruction count is smaller after removal of the unused
   pipeline branch; this is not by itself evidence of faster execution.
5. The original planner source, with its original parameter layout and disabled
   pipeline field retained, was compared against the submitted planner. Four
   invocations each verified 2,048 randomized task tables and packets: all matched,
   including predicted finish times (maximum delta **0 microseconds**). ABBA CUDA
   event measurements over T=1..8 gave median kernel-time changes from -0.95% to
   +1.22%; both binaries had occasional timing spikes. This synthetic kernel test
   does not establish whole-model performance.
6. All code-section byte multisets matched in 17 CPU expert-library object files
   from the retained old build and fresh build, including AVX2 experts and the
   worker pool. This is an object-code cross-check, not a claim that the complete
   executables or every linked address are identical.
7. A relocation-masked whole-body search uniquely located 12 IQ3_S AVX2 math
   functions in both frozen executables. All 12 retain the same 32-byte and
   64-byte alignment, despite changed absolute RVAs. Thus a changed JCC boundary
   alignment of those located functions does not explain this comparison. This
   does not exhaustively check every host function or all instruction-cache effects.

The reported `stage` timer accumulates host staging and PLE gather work. It can
overlap or be included in other intervals and must not be called graph capture
time or added as an independent component of total latency.

## Predeclared complete-model confirmation

An initial attempt at this confirmation was stopped after 18 completed requests:
the added telemetry harness accidentally launched a monitor after every request.
The **entire attempt is INVALID_HARNESS**, independent of its measured speeds,
and is retained in [invalid-harness-runs.csv](invalid-harness-runs.csv). None of
those timings enters acceptance. The corrected harness verified a real monitor
lifecycle before restarting the complete protocol and checks for exactly one
monitor before every request. Engine code and tuning were unchanged.

Same RTX 5080 + RTX 2070, i7-9700K, Windows, driver 610.88, CUDA 13.0 and Swift
IQ3_S. The original and submitted executables have the same hashes as the earlier
comparison. Both use the same expert profile, 3,788 primary and 3,200 helper slots,
five CPU workers, MTP 6, int8 KV, 262,144 context limit, adaptive-cache settings,
8,202 input tokens, seed 2026110492, temperature 1, top-p 0.95 and top-k 20.

Before timing, this batch was fixed at 12 process reloads in the order
old1/new1/new2/old2/new3/old3/old4/new4/old5/new5/new6/old6. Each reload has one
excluded 2,048-token warmup, five measured continuous 2,048-token requests and
one separate natural-completion answer check. There is no KV reuse. All **84**
request records are retained in [runs.csv](runs.csv); no sample extension is
allowed just to obtain a pass. GPU clocks, power, temperature and utilization
were sampled every five seconds. No compiler or profiler ran alongside timing.

Acceptance requires new aggregate throughput at least as high as old, and the
one-sided 95% lower ratio at least 0.97. This time the bootstrap resamples the
**six paired process reloads**, not individual requests. All 6^6 = 46,656 possible
paired block resamples are evaluated. The guard excludes a material regression
at the stated margin; it does not prove exact equality or every workload.

| Metric | Original tested build | Submitted source build |
| --- | ---: | ---: |
| Measured decode requests | 30 | 30 |
| Aggregate decode, token/s | 97.413 | 98.472 |
| Per-request range, token/s | 86.479–111.506 | 87.606–112.329 |
| Decode coefficient of variation | 5.23% | 7.32% |
| Aggregate prefill, token/s | 1558.69 | 1569.20 |
| Mean request wall time, seconds | 26.313 | 26.052 |
| Mean time to first output, seconds | 5.345 | 5.309 |
| Draft acceptance | 89.02% | 88.57% |
| Primary cold-weight MB/output token | 30.614 | 30.048 |
| Verification windows | 13699 | 13611 |
| Aggregate decode ms/verification window | 46.041 | 45.841 |

New aggregate change: **+1.09%**. One-sided 95% lower speed ratio:
**0.9789**; two-sided 95% interval:
**[0.9673, 1.0519]**.

| Reload pair | Old token/s | New token/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 100.652 | 92.749 | -7.85% |
| 2 | 97.145 | 98.296 | +1.18% |
| 3 | 98.943 | 100.156 | +1.23% |
| 4 | 95.186 | 97.737 | +2.68% |
| 5 | 95.276 | 104.269 | +9.44% |
| 6 | 97.506 | 98.332 | +0.85% |

## Gates and limits

- Complete answers: **PASS**, 12 natural completions. All eight recalled
  values, all 128 prefix sums and total 8256 are checked; complete JSON objects
  are canonicalized and compared. Canonical hashes: `e015d42f85d733468b7ff14529813555697321828467c7f93e0c6f9843b41c3c`.
- Decode non-regression gate: **PASS**.
- Main merge gate: **PASS**.
- Every submitted-build sample above 100 token/s with CV below 5%:
  **NOT_PASS**. Average throughput does not imply a per-request floor.
- Formal launch files, formal executable and formal expert profile: hashes
  unchanged; the ordinary formal service was restored after the batch.

No speculative code change was introduced based only on the initial 2% gap.
The slowdown did not reproduce consistently in this predeclared confirmation:
five of six reload pairs favored the submitted build, while one favored the
original. This is consistent with substantial run and generation-path variation;
it is not proof that every source path has exactly equal cost.
The fixed-token negative timing result and the earlier rejected batch remain
visible above. Numerical equivalence and kernel checks narrow the possible cause
but cannot alone prove zero timing impact from all host-code or system effects.
The measured complete-model gate, with its explicit margin and uncertainty, is
the release decision. Default-path broad quality, other models, long contexts,
HIP, SYCL and Metal remain outside this validation.

Raw requests, complete answers, reasoning, launch files, full disassembly and
telemetry remain in the local evidence directories. Model data, local absolute
paths, executable binaries and raw profiler captures are not published here.
