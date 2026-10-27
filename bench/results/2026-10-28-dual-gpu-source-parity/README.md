# Full-model comparison of the extracted scheduler

2026-10-28. Acceptance result: **NOT_PASS**. The tested source commit is
`2f8c82bed82a77deb2b5f51ca348848e4fb2ea24`. No engine code was changed during this comparison.

The selected experimental executable and the fresh source build ran on the same
Windows PC: RTX 5080 + RTX 2070, i7-9700K, NVIDIA driver 610.88, CUDA 13.0,
MSVC 14.44.35207, Release, CUDA architectures 75 and 120. Both used Swift IQ3_S
and the opt-in configuration in [the reference preset](../../../docs/examples/dual-gpu-moe-5080-2070.json).
The full original environment was passed to both; removed experimental flags
were disabled or equivalent to their retained defaults. Each executable used
its matching `strata_kq.dll`. The DLL CPU text and CUDA fatbin sections were
byte-identical, although the complete DLL hashes differed.

## Protocol

Each process reload started from the same immutable expert profile. The effective
model, cache capacities (3,788 primary / 3,200 helper expert slots), five CPU
workers, MTP 6, int8 KV, context limit 262,144 and cache settings matched. Free
VRAM at startup varied slightly. Every request had 8,202 input tokens and zero KV
reuse. Temperature was 1, top-p 0.95, top-k 20, seed 2026110492.

Each reload ran one excluded 2,048-token warmup, five measured continuous
2,048-token requests, then a separate natural-completion answer check. Timed
requests used reasoning budget 0 to retain the established continuous-decode
protocol; they are length-limited throughput samples, not complete-answer tests.
The separate complete-answer requests used reasoning budget 24,576 and maximum
32,768 output tokens. No requests failed or were discarded.

The initial order was old/new/new/old. Its lower new-build aggregate triggered
one bounded supplementary batch in new/old/old/new order. The protocol and
thresholds were unchanged. All 40 timed requests, eight warmups and eight answer
checks are retained in [runs.csv](runs.csv); the first negative batch is included.
All input/profile hashes remained unchanged. The ordinary formal service was
restored after each batch, and its protected file hashes were unchanged.

## Measured results

| Metric | Original tested build | Submitted source build |
| --- | ---: | ---: |
| Measured decode requests | 20 | 20 |
| Aggregate decode, token/s | 98.560 | 96.530 |
| Per-request decode range, token/s | 82.001–111.681 | 90.194–102.722 |
| Decode coefficient of variation | 7.31% | 3.52% |
| Aggregate prefill, token/s | 1526.73 | 1522.01 |
| Mean timed-request wall time, seconds | 26.172 | 26.626 |
| Mean time to first output, seconds | 5.453 | 5.466 |
| Primary cold weight MB/output token | 29.675 | 30.388 |
| Helper cold weight MB/output token | 0.551 | 0.565 |
| Draft acceptance | 89.57% | 88.74% |

Aggregate throughput is total output tokens divided by total decode time.
The new-build aggregate difference was **-2.06%**.

| Batch | Old token/s | New token/s | Difference |
| --- | ---: | ---: | ---: |
| initial | 99.156 | 95.678 | -3.51% |
| supplementary | 97.971 | 97.397 | -0.59% |

## Acceptance and limits

- Full-answer correctness: **PASS**, 8/8 natural completions. All
  eight recalled values, all 128 prefix sums and total 8256 were exact. The
  complete JSON objects were equal after canonicalization. Their canonical
  SHA-256 was `e015d42f85d733468b7ff14529813555697321828467c7f93e0c6f9843b41c3c`.
- Raw text/float/logit bitwise parity: **NOT_ESTABLISHED**. Answer whitespace and
  reasoning traces varied; equal answers are not evidence of identical logits.
- Decode non-regression gate: **NOT_PASS**.
  Before timing, acceptance required a new aggregate at least as high as the
  original and a paired request-bootstrap one-sided 95% lower speed ratio of at
  least 0.97. The observed lower ratio was
  0.9509; the exploratory
  two-sided 95% interval was [0.9454, 1.0146]. This request-level bootstrap
  has only four process reloads per build and followed one supplementary batch;
  it is not proof of zero regression or a broad release confidence guarantee.
- Every new-build request above 100 token/s with CV below 5%:
  **NOT_PASS**.
- Main-branch merge gate: **NOT_PASS**. The evidence does not establish no performance regression; main must remain unchanged.

Cold-weight traffic, generated reasoning, draft acceptance and graph preparation
cost varied between requests. These are measured differences, not proof of the
root cause. No speculative engine change was made to turn this acceptance test
into a pass. Historical 102.74 token/s measurements must not replace this fresh
same-day comparison. Broader quality, default-path full-model parity, other
models, long-context limits and HIP/SYCL/Metal remain outside this test.

See [summary.json](summary.json) for all aggregate/session statistics and
[identity.json](identity.json) for executable, input and profile hashes. Raw
requests, complete outputs, service logs and restoration checks remain in the
local acceptance evidence directory. This is a local controlled comparison,
not an independently reproduced or standardized community benchmark.
