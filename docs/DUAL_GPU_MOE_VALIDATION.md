# Validation of the extracted scheduler source

Date: 2026-10-28. This record covers the source extraction, focused tests and
subsequent full-model comparison. The earlier experimental performance
measurements are in [the scheduler notes](DUAL_GPU_MOE_SCHEDULER.md).

## Build identity

- Base: `1aff6d14117b88171fcfcec8f2b55fdc47f2b165`, engine 0.1.50.
- Windows x64, MSVC 14.44.35207, CUDA 13.0, Release/Ninja, architectures 75 and 120.
- RTX 5080 and RTX 2070, NVIDIA driver 610.88; i7-9700K CPU.
- Fresh build directory; no experimental object files or libraries reused.
- Native experts, portable AVX2 CPU baseline, Q6K experts, MMQ K-quants and TQ
  overlay enabled; HIP, SYCL, DSV41, MMQ TAB and prefill MMQ disabled.
- GGML source snapshot: 1,389 `ggml/` and `cmake/` file blobs verified exactly
  against the repository's pinned llama.cpp revision
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. No dependency changes are part of
  this commit.
- Built `strata.exe` SHA-256:
  `1a26de782272867d3dbca1cbcca5b2991e4c444f63a7ed4c3848917001e3b4a6`.

## Results

| Check | Result |
| --- | --- |
| Clean CUDA build and final dependency rebuild | PASS |
| `gpu_task_reuse_test 0` and `gpu_task_reuse_test 1` | PASS: 462 captured replays per GPU, 924 total; full weight patterns, protected hits, empty copies, canaries, helper ownership and tail policy |
| Compute Sanitizer `memcheck --error-exitcode 99` on both reuse tests | PASS: 0 errors on each GPU |
| `host_vmm_test` | PASS: three allocation/free lifetimes, 262,267 page reads per lifetime, helper DMA across a chunk boundary, pin-cap rejection and ordinary allocation fallback |
| `pool_test --selftest` with new feature options unset | PASS |
| `iq_avx2_parity` with new feature options unset | PASS: 0 failures |
| `iq_avx2_parity` with `STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_IQ256_EXACT_MODE=2` | PASS: 0 failures |
| Local precise-solver reference versus extracted GPU planner | PASS: 2,048 randomized route/task tables identical; largest predicted-time difference 0.000244 microseconds |
| `cpu_expert_publish_test <compatible GGUF>` with ready DAG, per-expert publication, serial verification and changing-input checks enabled | PASS: 3,240 concurrent GPU-consumer checks, generation resets and guard rows; sampled bitwise serial checks passed |
| `strata --help` from the fresh build | PASS |
| Staged whitespace and content audit | PASS: no local absolute paths, credentials, model files, binaries or raw profiler captures |
| Same-day full-model answer comparison against the selected experimental executable | PASS: eight natural completions; exact expected values and identical canonical JSON objects |
| Initial same-day decode comparison | NOT_PASS: 20 measured requests per executable; original 98.560 token/s, extracted build 96.530 token/s, -2.06%; merge was withheld |
| Subsequent predeclared full-model confirmation | PASS at the stated margin: 30 measured requests per executable; original 97.413 token/s, extracted build 98.472 token/s, +1.09%; paired reload-bootstrap one-sided 95% lower ratio 0.9789 versus required 0.97 |
| Subsequent complete-answer checks | PASS: 12/12 natural completions; the same exact canonical JSON as the initial comparison |

The CPU publication test used real Swift IQ3_S expert weights from layers 3,
20 and 47, group counts 1/4/8/16/32/80 and changing activations. It ran with
`STRATA_CPU_EXPERT_DAG=1`, `STRATA_CPU_DAG_READY=1`,
`STRATA_CPU_EXPERT_PUBLISH=1`, `STRATA_CPU_EXPERT_DAG_VERIFY=1`,
`STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_IQ256_EXACT_MODE=2`, and
`BENCH_READY_VERIFY=1`. Model files and raw output are kept outside the repository.
Diagnostic microsecond timings are not end-to-end performance measurements.

The precise-solver oracle was the retained pre-optimization diagnostic used in
the original experiments. It is a local validation artifact, not a second
production scheduler. The reusable regression tests are included in this commit.

## Full-model comparison

The tested source revision was `2f8c82bed82a77deb2b5f51ca348848e4fb2ea24`,
using the fresh executable hash above. The selected original executable was
`9860f56b4d8a997a591b578752594fb8cd9bf94372b4d9e4f586bff516f52ca6`.
The two builds ran the same 8,202-input / 2,048-output protocol, initial expert
profile, environment, effective cache capacities and sampler settings. Eight
process reloads used old/new/new/old followed by new/old/old/new order; each
reload excluded one warmup and retained five measured requests. Separate
complete-answer checks all passed. Text formatting and reasoning traces varied;
this is not a bitwise logit-parity claim.

The first batch measured -3.51% and the supplementary batch -0.59%; the combined
new-build difference was -2.06%. This does not establish no performance
regression, and the main-branch merge gate did not pass. Cold-weight traffic and
generation traces varied; the cause is not established. The formal service was
restored after both batches, with protected file hashes unchanged.

The [full report and all 56 request records](../bench/results/2026-10-28-dual-gpu-source-parity/README.md)
include warmups, timing samples, complete-answer checks, variability and identity
hashes. No engine code was changed during this acceptance comparison.

## Follow-up investigation and acceptance

The [follow-up investigation](../bench/results/2026-10-28-dual-gpu-regression-investigation/README.md)
retains both negative timing observations and the complete confirmation data.
Five fixed-token runs had identical residual hashes and argmax records for all
456 windows. On each GPU architecture, 21 of 22 selected profiled kernel variants
had identical machine words; the changed planner produced identical task tables
and finish-time estimates in 2,048 randomized cases. CPU expert object code and
the checked linked math-function alignments matched. No production source change
was made in this investigation.

The predeclared confirmation used six process reloads and 30 timing samples per
binary, plus 12 excluded warmups and 12 separate complete-answer checks. It
passed the unchanged throughput/margin gate after the original slowdown failed
to reproduce consistently. This is a bounded non-regression decision, not proof
of exactly zero performance difference. The full bootstrap interval, fixed-token
-1.63% diagnostic, earlier -2.06% result, and an invalid telemetry-harness attempt
are all disclosed. The formal service, protected hashes and tracked CPU affinities
were restored. The new-build range was 87.606–112.329 token/s: stable 100 token/s
on every request is **NOT_PASS**.

## Unverified scope

- HIP, gfx906, SYCL and Metal builds/runtime: **NOT_RUN**. CUDA guards do not
  substitute for those builds. No upstream ready-for-review PR is claimed.
- An extra CUDA 12.2 / SM61 compile attempt failed the installed MSVC STL
  compatibility gate (`STL1002`, requiring CUDA 12.4 or newer). It did not reach
  runtime validation. CUDA 12.x compatibility is **NOT_VALIDATED**; no check was
  bypassed and no toolchain was installed or changed.
- Default-path end-to-end bitwise parity, broad quality, other models and
  long-context acceptance: **NOT_RUN**. The bounded full-model comparison above
  does not establish these. The historical 102.74 token/s aggregate belongs to
  the explicitly identified earlier executable, not this build.
- The formal service, its engine, launch files and expert profile were not
  replaced. This source commit is not a deployment or a binary release.
