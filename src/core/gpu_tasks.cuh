// CUDA implementation included by remote_expert_opt.cu, an existing CUDA target.
#include "strata/core/gpu_tasks.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/cpu/hadamard.hpp"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
namespace strata::core {
namespace {
// A bounded layer makespan model. CPU cost is the whole worker pool's service
// time (not a single worker), measured separately for each byte-size family.
// It is a prediction, not a claim that independent RAM bandwidths add together.
__device__ float schedule_finish(const GpuTaskPlan &a, const int *first, const int *kinds, const int *rows,
                                 float *finish, int *allocated) {
    float fetch[2] = {0, 0};
    finish[0] = 0;
    finish[1] = a.cost.helper_start;
    finish[2] = 0;
    allocated[0] = allocated[1] = 0;
    float returns = 0;
    for (int j = 0; j < a.n; ++j)
        if (first[j] == j) {
            const int k = kinds[j];
            if (k == 0 || k == 2) {
                const int d = k == 2;
                finish[d] += a.cost.group[d] + rows[j] * a.cost.row[d];
                if (d)
                    returns += rows[j] * a.cost.helper_return;
            } else if (k == -1) {
                if (finish[2] == 0)
                    finish[2] = a.cost.cpu_start;
                finish[2] += a.cost.cpu_group + rows[j] * a.cost.cpu_row;
            }
        }
    for (int j = 0; j < a.n; ++j)
        if (first[j] == j && (kinds[j] == 1 || kinds[j] == 3)) {
            const int d = kinds[j] == 3;
            ++allocated[d];
            fetch[d] += a.cost.fetch[d];
            finish[d] = max(finish[d], fetch[d]) + a.cost.group[d] + rows[j] * a.cost.row[d];
            if (d)
                returns += rows[j] * a.cost.helper_return;
        }
    finish[1] += returns;
    return max(max(finish[0], finish[1]), finish[2]);
}
struct QueueCost {
    int groups[5] = {}, rows[5] = {};
};
__device__ __forceinline__ void queue_change(QueueCost &q, int kind, int rows, int delta) {
    // Fixed indices let the compiler keep the five small queues in registers.
    switch (kind) {
    case 0:
        q.groups[0] += delta;
        q.rows[0] += delta * rows;
        break;
    case 1:
        q.groups[1] += delta;
        q.rows[1] += delta * rows;
        break;
    case 2:
        q.groups[2] += delta;
        q.rows[2] += delta * rows;
        break;
    case 3:
        q.groups[3] += delta;
        q.rows[3] += delta * rows;
        break;
    case -1:
        q.groups[4] += delta;
        q.rows[4] += delta * rows;
        break;
    }
}
__device__ __forceinline__ float contention_finish(const GpuTaskPlan &a, const QueueCost &q, float *finish) {
    // Approximate division is confined to scheduling estimates, never model math.
    const auto &c = a.cost;
    const float cpu = q.groups[4] ? c.cpu_start + q.groups[4] * c.cpu_group + q.rows[4] * c.cpu_row : 0;
    const float memory = q.groups[4] * c.cpu_ram;
    const float duty = min(1.f, __fdividef(memory, max(cpu - c.cpu_start, 1.f)));
    const float bare0 = q.groups[1] * c.fetch[0], bare1 = q.groups[3] * c.fetch[1];
    float f0 = bare0, f1 = bare1, ct = cpu;
    // Two bounded overlap refinements. CPU compute time is not charged again
    // as memory traffic; only its measured memory-service component stretches.
    for (int it = 0; it < 2; ++it) {
        const float u0 = bare0 * (1 + c.main_cpu * duty * __fdividef(min(f0, ct), max(f0, 1.f)) +
                                  c.gpu_cross * __fdividef(min(f0, f1), max(f0, 1.f)));
        const float u1 = bare1 * (1 + c.helper_cpu * duty * __fdividef(min(f1, ct), max(f1, 1.f)) +
                                  c.gpu_cross * __fdividef(min(f0, f1), max(f1, 1.f)));
        const float uc = cpu + memory * (c.cpu_main * __fdividef(min(ct, f0), max(ct, 1.f)) +
                                         c.cpu_helper * __fdividef(min(ct, f1), max(ct, 1.f)));
        f0 = u0;
        f1 = u1;
        ct = uc;
    }
    const float hot0 = q.groups[0] * c.group[0] + q.rows[0] * c.row[0];
    const float hot1 = q.groups[2] * c.group[1] + q.rows[2] * (c.row[1] + c.helper_return);
    const float cold0 = q.groups[1] * c.group[0] + q.rows[1] * c.row[0];
    const float cold1 = q.groups[3] * c.group[1] + q.rows[3] * (c.row[1] + c.helper_return);
    finish[0] = max(hot0, f0) + cold0;
    finish[1] = c.helper_start + max(hot1 + cold1, f1 + (q.groups[3] ? __fdividef(cold1, q.groups[3]) : 0.f));
    finish[2] = ct;
    const float shared_floor = (q.groups[1] + q.groups[3] + q.groups[4]) * c.cpu_ram;
    return max(max(max(finish[0], finish[1]), finish[2]), shared_floor);
}
__global__ void plan_gpu_tasks(GpuTaskPlan a) {
    __shared__ int ids[80], first[80], kinds[80], candidate[80];
    __shared__ GpuExpertAddress address[80];
    __shared__ int cache_index[80], reuse_rows[80];
    __shared__ GpuTaskPacket packet;
    auto *words = (uint32_t *)&packet;
    for (int j = threadIdx.x; j < sizeof(packet) / 4; j += blockDim.x)
        words[j] = 0;
    const int i = threadIdx.x;
    if (i < a.n)
        ids[i] = a.ids[i];
    __syncthreads();
    if (i == 0) {
        packet.transfer.input = a.packet->transfer.input;
        packet.transfer.output = a.packet->transfer.output;
        packet.transfer.ready = a.packet->transfer.ready;
        packet.transfer.verify = a.packet->transfer.verify;
        packet.transfer.sequence = a.sequence;
    }
    if (i < a.n) {
        const int e = ids[i];
        int f = i, mr = i % a.k;
        for (int j = 0; j < a.n; ++j)
            if (ids[j] == e) {
                f = min(f, j);
                mr = min(mr, j % a.k);
            }
        first[i] = f;
        int reuse = 0;
        if (a.cold_cache)
            for (int j = 0; j < a.n; ++j)
                reuse += ids[j] == e;
        reuse_rows[i] = reuse;
        cache_index[i] = -1;
        address[i] = e >= 0 && e < a.experts ? a.addresses[e] : GpuExpertAddress{};
        // Temporary reuse must preserve the original helper ownership and tail policy.
        if (a.cold_cache && !address[i].primary && !address[i].remote && (a.tail_rank <= 0 || mr < a.tail_rank) &&
            e >= 0 && e < a.experts)
            for (int s = 0; s < a.cold_cache_slots; ++s)
                if (a.cold_cache[s].expert == e) {
                    address[i].primary = a.cold_cache[s].pointer;
                    cache_index[i] = s;
                    break;
                }
        int kd = -1;
        if (address[i].primary)
            kd = 0;
        else if (address[i].remote)
            kd = 2;
        else if (a.tail_rank > 0 && mr >= a.tail_rank)
            kd = -2;
        kinds[i] = kd;
    }
    __syncthreads();
    if (i < a.n) {
        int misses = 0, rank = 0;
        for (int j = 0; j < a.n; ++j)
            if (first[j] == j && kinds[j] == -1) {
                ++misses;
                rank += j < i;
            }
        int m = (misses * a.pcie_num) >> 8;
        if (a.min_one && a.pcie_num > 0 && misses >= 2)
            m = max(m, 1);
        candidate[i] = first[i] == i && kinds[i] == -1 && rank >= misses - m && address[i].mapped != 0;
    }
    __syncthreads();
    if (i == 0 && !a.joint_schedule) {
        int allocated[2] = {0, 0};
        float fetch[2] = {0, 0}, compute[2] = {0, a.cost.helper_start}, returns = 0;
        for (int j = 0; j < a.n; ++j)
            if (first[j] == j && (kinds[j] == 0 || kinds[j] == 2)) {
                const int card = kinds[j] == 2;
                int rows = 0;
                for (int q = 0; q < a.n; ++q)
                    rows += first[q] == j;
                compute[card] += a.cost.group[card] + rows * a.cost.row[card];
                if (card)
                    returns += rows * a.cost.helper_return;
            }
        // Account for primary-only misses before placing flexible jobs. This
        // prevents an early flexible job from taking the primary while its
        // later, primary-only queue would have hidden a helper transfer.
        for (int pass = 0; pass < 2; ++pass)
            for (int j = 0; j < a.n; ++j)
                if (candidate[j] && int(address[j].helper_mapped != 0) == pass) {
                    int rows = 0;
                    for (int q = 0; q < a.n; ++q)
                        rows += first[q] == j;
                    const float p =
                        max(compute[0], fetch[0] + a.cost.fetch[0]) + a.cost.group[0] + rows * a.cost.row[0];
                    const float h =
                        max(compute[1], fetch[1] + a.cost.fetch[1]) + a.cost.group[1] + rows * a.cost.row[1];
                    const float p_span = max(p, compute[1] + returns);
                    const float h_span = max(compute[0], h + returns + rows * a.cost.helper_return);
                    const bool helper = a.helper_cold_cap > 0 && address[j].helper_mapped &&
                                        allocated[1] < a.helper_cold_cap &&
                                        (allocated[0] >= a.staging_cap || h_span < p_span);
                    const int card = helper ? 1 : 0;
                    if (helper || allocated[0] < a.staging_cap) {
                        kinds[j] = helper ? 3 : 1;
                        ++allocated[card];
                        fetch[card] += a.cost.fetch[card];
                        compute[card] = helper ? h : p;
                        if (helper)
                            returns += rows * a.cost.helper_return;
                    }
                }
        packet.predicted_us[0] = compute[0];
        packet.predicted_us[1] = compute[1] + returns;
    }
    if (i == 0 && a.joint_schedule == 1) {
        int rows[80] = {};
        float finish[3];
        int used[2];
        for (int j = 0; j < a.n; ++j)
            ++rows[first[j]];
        // Linear greedy descent, not exhaustive search: start with all cold
        // work on CPU and move it only when the whole layer finishes earlier.
        // Helper residency is revisited after the cold queues have been placed.
        schedule_finish(a, first, kinds, rows, finish, used);
        float fetch[2] = {0, 0};
        int cpu_groups = 0;
        for (int j = 0; j < a.n; ++j)
            cpu_groups += first[j] == j && kinds[j] == -1;
        for (int pass = 0; pass < 2; ++pass)
            for (int j = 0; j < a.n; ++j)
                if (first[j] == j && kinds[j] == (pass == 0 ? -1 : 2)) {
                    const float cpu_work = a.cost.cpu_group + rows[j] * a.cost.cpu_row;
                    const float helper_work = a.cost.group[1] + rows[j] * (a.cost.row[1] + a.cost.helper_return);
                    const float cpu_after = pass == 0 ? (cpu_groups == 1 ? 0 : finish[2] - cpu_work)
                                                      : finish[2] + cpu_work + (cpu_groups == 0 ? a.cost.cpu_start : 0);
                    const float helper_after = finish[1] - (pass == 1 ? helper_work : 0);
                    const float current = max(max(finish[0], finish[1]), finish[2]);
                    float best = current;
                    int choice = pass == 0 ? -1 : 2;
                    float pfinish = finish[0], hfinish = finish[1];
                    if (pass == 1 && max(max(finish[0], helper_after), cpu_after) + 0.1f < best) {
                        best = max(max(finish[0], helper_after), cpu_after);
                        choice = -1;
                    }
                    if (a.pcie_num > 0 && address[j].mapped && used[0] < a.staging_cap) {
                        const float proposed =
                            max(finish[0], fetch[0] + a.cost.fetch[0]) + a.cost.group[0] + rows[j] * a.cost.row[0];
                        const float span = max(max(proposed, helper_after), pass == 0 ? cpu_after : finish[2]);
                        if (span + 0.1f < best) {
                            best = span;
                            choice = 1;
                            pfinish = proposed;
                        }
                    }
                    if (pass == 0 && a.pcie_num > 0 && address[j].helper_mapped && used[1] < a.helper_cold_cap) {
                        const float proposed = max(finish[1], fetch[1] + a.cost.fetch[1]) + helper_work;
                        const float span = max(max(finish[0], proposed), cpu_after);
                        if (span + 0.1f < best) {
                            best = span;
                            choice = 3;
                            hfinish = proposed;
                        }
                    }
                    if (best + 0.1f >= current)
                        continue;
                    kinds[j] = choice;
                    if (pass == 0) {
                        --cpu_groups;
                        finish[2] = cpu_after;
                    } else if (choice == -1) {
                        ++cpu_groups;
                        finish[2] = cpu_after;
                    }
                    finish[1] = helper_after;
                    if (choice == 1) {
                        ++used[0];
                        fetch[0] += a.cost.fetch[0];
                        finish[0] = pfinish;
                    }
                    if (choice == 3) {
                        ++used[1];
                        fetch[1] += a.cost.fetch[1];
                        finish[1] = hfinish;
                    }
                }
        schedule_finish(a, first, kinds, rows, finish, used);
        for (int d = 0; d < 3; ++d)
            packet.predicted_us[d] = finish[d];
    }
    if (i == 0 && a.joint_schedule == 2) {
        int rows[80] = {}, max_reuse = 0;
        QueueCost q;
        float finish[3];
        for (int j = 0; j < a.n; ++j)
            ++rows[first[j]];
        for (int j = 0; j < a.n; ++j)
            if (first[j] == j && kinds[j] >= -1) {
                queue_change(q, kinds[j], rows[j], 1);
                max_reuse = max(max_reuse, rows[j]);
            }
        float current = contention_finish(a, q, finish);
        // High-reuse experts first: GPU fetch is amortized over their rows.
        // Two sweeps, bounded by 80 experts; no exponential assignment search.
        for (int pass = 0; pass < 2; ++pass)
            for (int reuse = max_reuse; reuse >= 1; --reuse)
                for (int j = 0; j < a.n; ++j)
                    if (first[j] == j && rows[j] == reuse && kinds[j] == (pass == 0 ? -1 : 2)) {
                        const int original = kinds[j];
                        int choice = original;
                        float best = current;
                        QueueCost removed = q;
                        queue_change(removed, original, rows[j], -1);
                        for (int option = 0; option < 3; ++option) {
                            const int kind = option == 0 ? -1 : option == 1 ? 1 : (pass == 0 ? 3 : 2);
                            if (kind == original)
                                continue;
                            if (kind == 1 &&
                                (a.pcie_num <= 0 || !address[j].mapped || removed.groups[1] >= a.staging_cap))
                                continue;
                            if (kind == 3 && (a.pcie_num <= 0 || !address[j].helper_mapped ||
                                              removed.groups[3] >= a.helper_cold_cap))
                                continue;
                            QueueCost candidate = removed;
                            queue_change(candidate, kind, rows[j], 1);
                            const float span = contention_finish(a, candidate, finish);
                            if (span + 1.f < best) {
                                best = span;
                                choice = kind;
                            }
                        }
                        if (choice != original) {
                            q = removed;
                            queue_change(q, choice, rows[j], 1);
                            kinds[j] = choice;
                            current = best;
                        }
                    }
        contention_finish(a, q, finish);
        for (int d = 0; d < 3; ++d)
            packet.predicted_us[d] = finish[d];
    }
    if (i == 0 && a.cold_cache) {
        const uint32_t stamp = ++*a.cold_clock;
        bool used[4] = {};
        int cold_first[80], groups = 0;
        for (int j = 0; j < a.n; ++j)
            if (first[j] == j) {
                if (cache_index[j] >= 0 && kinds[j] == 0) {
                    const int slot = cache_index[j];
                    used[slot] = true;
                    a.cold_cache[slot].last_use = stamp;
                    ++packet.reuse_groups;
                    packet.reuse_entries += reuse_rows[j];
                    ++a.cold_stats[0];
                    a.cold_stats[1] += reuse_rows[j];
                    a.cold_stats[4] += a.blob_bytes;
                }
                if (kinds[j] == 1) {
                    cold_first[groups] = j;
                    a.cold_plan->destination[groups] = a.cold_stage + groups * a.blob_bytes;
                    a.cold_plan->slot[groups] = -1;
                    a.cold_plan->expert[groups] = ids[j];
                    ++groups;
                }
            }
        // Admit only weights already assigned to the primary copy queue, with
        // high row reuse first. No speculative RAM traffic or current-hit eviction.
        for (int pick = 0; pick < a.cold_cache_slots; ++pick) {
            int slot = -1, best = -1;
            uint32_t age = 0;
            for (int s = 0; s < a.cold_cache_slots; ++s)
                if (!used[s]) {
                    const uint32_t candidate = stamp - a.cold_cache[s].last_use;
                    if (slot < 0 || candidate > age) {
                        slot = s;
                        age = candidate;
                    }
                }
            if (slot < 0)
                break;
            for (int q = 0; q < groups; ++q)
                if (a.cold_plan->slot[q] < 0 && (best < 0 || reuse_rows[cold_first[q]] > reuse_rows[cold_first[best]]))
                    best = q;
            if (best < 0)
                break;
            used[slot] = true;
            if (a.cold_cache[slot].expert >= 0)
                ++a.cold_stats[3];
            a.cold_cache[slot].expert = -1;
            a.cold_plan->destination[best] = a.cold_cache[slot].pointer;
            a.cold_plan->slot[best] = slot;
        }
    }
    __syncthreads();
    int *start = a.plan + 4;
    int *dst = start + a.cap + 1;
    int *tok = dst + a.cap;
    const int po = (4 + a.cap + 1 + 2 * a.cap + 1) & ~1;
    auto *ptr = (unsigned long long *)(a.plan + po);
    auto *ptr2 = ptr + a.cap;
    int *start2 = a.plan + po + 4 * a.cap;
    if (i < a.n) {
        const int f = first[i], kd = kinds[f];
        int group = 0, entry = 0, primary_entries = 0, remote_entries = 0;
        for (int j = 0; j < a.n; ++j) {
            const int other = kinds[first[j]];
            primary_entries += other == 0;
            remote_entries += other == 2;
            if (other == kd) {
                group += first[j] == j && j < f;
                entry += first[j] < f || (first[j] == f && j < i);
            }
        }
        packet.kind[i] = kd;
        packet.cpu_group[i] = kd == -1 ? group : -1;
        if (kd == -1 && i == f)
            packet.cpu_expert[group] = ids[i];
        if (kd == 0 || kd == 1) {
            const int e = entry + (kd == 1 ? primary_entries : 0);
            dst[e] = i;
            tok[e] = i / a.k;
            if (i == f) {
                if (kd == 0) {
                    ptr[group] = address[i].primary;
                    start[group] = e;
                } else {
                    ptr2[group] = address[i].mapped;
                    start2[group] = e;
                }
            }
        } else if (kd == 2 || kd == 3) {
            auto &meta = kd == 2 ? packet.remote : packet.cold;
            const int dest = entry + (kd == 3 ? remote_entries : 0);
            meta.dst[entry] = dest;
            meta.tok[entry] = i / a.k;
            packet.transfer.row[dest] = i;
            if (i == f) {
                meta.ptr[group] = kd == 2 ? address[i].remote : address[i].helper_mapped;
                meta.start[group] = entry;
            }
        }
    }
    if (i == 0) {
        int g0 = 0, g1 = 0, g2 = 0, g3 = 0, e0 = 0, e1 = 0, e2 = 0, e3 = 0, cpu = 0, skipped = 0, error = 0;
        for (int j = 0; j < a.n; ++j) {
            const int kd = kinds[first[j]];
            e0 += kd == 0;
            e1 += kd == 1;
            e2 += kd == 2;
            e3 += kd == 3;
            error |= ids[j] < 0 || ids[j] >= a.experts;
            if (first[j] == j) {
                g0 += kd == 0;
                g1 += kd == 1;
                g2 += kd == 2;
                g3 += kd == 3;
                cpu += kd == -1;
                skipped += kd == -2;
            }
        }
        start[g0] = e0;
        start2[g1] = e0 + e1;
        a.plan[0] = g0;
        a.plan[1] = e0 + e1;
        a.plan[2] = g1;
        a.plan[3] = 0;
        packet.remote.count = g2;
        packet.remote.start[g2] = e2;
        packet.cold.count = g3;
        packet.cold.start[g3] = e3;
        packet.transfer.entries = e2 + e3;
        packet.cpu_groups = cpu;
        packet.skipped = skipped;
        packet.pcie_groups = g1;
        packet.error = error;
        for (int j = 0; j < 4; ++j)
            a.host_counts[j] = a.plan[j];
    }
    __syncthreads();
    const auto *source = (const uint32_t *)&packet;
    auto *target = (volatile uint32_t *)a.packet;
    for (int j = i; j < offsetof(GpuTaskPacket, ready) / 4; j += blockDim.x)
        target[j] = source[j];
    __threadfence_system();
    __syncthreads();
    if (i == 0) {
        *((volatile uint32_t *)a.flag_a) = a.sequence;
        *((volatile uint32_t *)a.flag_b) = a.sequence;
        __threadfence_system();
        *((volatile uint32_t *)&a.packet->ready) = a.sequence;
    }
}

__global__ void load_gpu_task(const GpuTaskPacket *packet, GpuRemoteMeta *meta, const volatile uint32_t *abort_flag,
                              uint32_t seq, GpuRemoteMeta *cold) {
    __shared__ int aborted;
    if (threadIdx.x == 0) {
        const auto *ready = (const volatile uint32_t *)&packet->ready;
        while (*ready != seq && *abort_flag != 0xffffffffu) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
        }
        aborted = *abort_flag == 0xffffffffu;
    }
    __syncthreads();
    auto *transfer = (RemoteTransferMeta *)(meta + 1);
    if (aborted) {
        if (threadIdx.x == 0) {
            meta->count = 0;
            meta->start[0] = 0;
            transfer->entries = 0;
            transfer->input = 0;
            if (cold) {
                cold->count = 0;
                cold->start[0] = 0;
            }
        }
        return;
    }
    // Volatile reads bypass any value retained from the previous window.
    const auto *src = (const volatile uint32_t *)&packet->remote;
    auto *dst = (uint32_t *)meta;
    for (int i = threadIdx.x; i < sizeof(GpuRemoteMeta) / 4; i += blockDim.x)
        dst[i] = src[i];
    if (cold) {
        const auto *csrc = (const volatile uint32_t *)&packet->cold;
        auto *cdst = (uint32_t *)cold;
        for (int j = threadIdx.x; j < sizeof(GpuRemoteMeta) / 4; j += blockDim.x)
            cdst[j] = csrc[j];
    }
    src = (const volatile uint32_t *)&packet->transfer;
    dst = (uint32_t *)transfer;
    for (int i = threadIdx.x; i < sizeof(RemoteTransferMeta) / 4; i += blockDim.x)
        dst[i] = src[i];
}
} // namespace
__global__ void fetch_cached_blobs_kernel(const unsigned long long *src, const int32_t *n, const GpuColdCachePlan *plan,
                                          long long per) {
    const long long total = (long long)*n * per;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long q = i / per, off = i - q * per;
        ((uint4 *)plan->destination[q])[off] = ((const uint4 *)src[q])[off];
    }
}
__global__ void commit_cached_blobs(unsigned long long *src, const int32_t *n, const GpuColdCachePlan *plan,
                                    GpuColdCacheEntry *cache, const uint32_t *clock, uint64_t *stats, uint64_t bytes) {
    const int q = threadIdx.x;
    if (q >= *n)
        return;
    src[q] = plan->destination[q];
    const int slot = plan->slot[q];
    if (slot >= 0) {
        cache[slot].last_use = *clock;
        cache[slot].expert = plan->expert[q];
        atomicAdd((unsigned long long *)(stats + 2), 1ull);
        atomicAdd((unsigned long long *)(stats + 5), (unsigned long long)bytes);
    }
}
void gpu_tasks_fetch_cached(unsigned long long *src, const int32_t *n, const GpuColdCachePlan *plan,
                            GpuColdCacheEntry *cache, const uint32_t *clock, uint64_t *stats, int64_t bytes, int blocks,
                            void *stream) {
    fetch_cached_blobs_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(src, n, plan, bytes / 16);
    commit_cached_blobs<<<1, 128, 0, (cudaStream_t)stream>>>(src, n, plan, cache, clock, stats, bytes);
}
void gpu_tasks_plan(const GpuTaskPlan &p, void *s) { plan_gpu_tasks<<<1, 128, 0, (cudaStream_t)s>>>(p); }
void gpu_tasks_load(const GpuTaskPacket *p, void *meta, const uint32_t *abort_flag, uint32_t seq, void *s, void *cold) {
    load_gpu_task<<<1, 128, 0, (cudaStream_t)s>>>(p, (GpuRemoteMeta *)meta, abort_flag, seq, (GpuRemoteMeta *)cold);
}

__global__ void prepare_helper_cold(GpuRemoteMeta *m, unsigned long long *src, int32_t *counts, uint8_t *weights,
                                    size_t stride, int cap) {
    const int q = threadIdx.x;
    if (q < cap) {
        counts[q] = q < m->count ? 1 : 0;
        counts[cap + q] = 0;
        src[q] = q < m->count ? m->ptr[q] : 0;
        if (q < m->count)
            m->ptr[q] = (unsigned long long)(weights + q * stride);
    }
}
__global__ void fetch_helper_expert(const unsigned long long *src, const int32_t *count, uint4 *dst, size_t words) {
    if (!*count)
        return;
    const auto *in = (const uint4 *)*src;
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < words; j += gridDim.x * blockDim.x)
        dst[j] = in[j];
}

#ifndef STRATA_TASK_PLANNER_TEST
// Publish only completed rows. Each cold expert has its own range; the next
// layer cannot reuse the packet/scratch until both helper branches have joined.
__global__ void scatter_completed_experts(const float4 *src, float4 *shadow, const RemoteTransferMeta *transfer,
                                          const GpuRemoteMeta *groups, int group) {
    if (group >= groups->count)
        return;
    const int begin = group < 0 ? 0 : groups->start[group];
    const int end = group < 0 ? groups->start[groups->count] : groups->start[group + 1];
    const int entry = begin + blockIdx.x;
    if (entry >= end)
        return;
    const int compact = groups->dst[entry], row = transfer->row[compact];
    auto *dst = (float4 *)transfer->output;
    for (int c = threadIdx.x; c < H / 4; c += blockDim.x) {
        const float4 v = src[(int64_t)compact * (H / 4) + c];
        dst[(int64_t)row * (H / 4) + c] = v;
        if (transfer->verify)
            shadow[(int64_t)compact * (H / 4) + c] = v;
    }
    if (transfer->ready) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0)
            ((volatile uint32_t *)transfer->ready)[row] = transfer->sequence;
    }
}
#if CUDART_VERSION >= 13000
__global__ void set_cold_count(const GpuRemoteMeta *meta, cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle, (unsigned)meta->count - 1u);
}
// The copy CTAs advance through active experts only. The last CTA for each
// expert makes that expert available to the compute branch independently.
__global__ void fetch_helper_active(const unsigned long long *src, const GpuRemoteMeta *meta, uint8_t *weights,
                                    size_t stride, size_t words, int32_t *completed) {
    for (int q = 0; q < meta->count; ++q) {
        const auto *in = (const uint4 *)src[q];
        auto *out = (uint4 *)(weights + q * stride);
        for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < words; j += gridDim.x * blockDim.x)
            out[j] = in[j];
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(completed + q, 1);
    }
}
__global__ void wait_helper_expert(const volatile int32_t *completed) {
    if (threadIdx.x == 0)
        while (*completed != 32)
            __nanosleep(128);
}
template <class Record>
bool capture_cold_switch(cudaStream_t stream, cudaGraphConditionalHandle handle, int cap, Record record,
                         std::string &err) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t *deps = nullptr;
    size_t count = 0;
    if (!check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, nullptr, &count), err))
        return false;
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeSwitch;
    params.conditional.size = cap;
    cudaGraphNode_t node = nullptr;
    if (!check(cudaGraphAddNode(&node, graph, deps, nullptr, count, &params), err))
        return false;
    cudaStream_t body = nullptr;
    if (!check(cudaStreamCreateWithFlags(&body, cudaStreamNonBlocking), err))
        return false;
    bool ok = true;
    for (int n = 1; n <= cap && ok; ++n) {
        ok = check(cudaStreamBeginCaptureToGraph(body, params.conditional.phGraph_out[n - 1], nullptr, nullptr, 0,
                                                 cudaStreamCaptureModeThreadLocal),
                   err);
        if (ok) {
            record(n, body);
            cudaGraph_t ended = nullptr;
            const auto recorded = cudaGetLastError();
            const auto end = cudaStreamEndCapture(body, &ended);
            ok = check(recorded, err) && check(end, err);
        }
    }
    cudaStreamDestroy(body);
    return ok &&
           check(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies), err);
}
struct ColdConditions {
    cudaGraphConditionalHandle fetch[16], compute[16];
};
__global__ void set_cold_conditions(const GpuRemoteMeta *meta, ColdConditions handles, int cap) {
    const int q = threadIdx.x;
    if (q < cap) {
        const unsigned active = q < meta->count;
        cudaGraphSetConditional(handles.fetch[q], active);
        cudaGraphSetConditional(handles.compute[q], active);
    }
}
template <class Record>
bool capture_cold_if(cudaStream_t stream, cudaGraphConditionalHandle handle, Record record, std::string &err) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t *deps = nullptr;
    size_t count = 0;
    if (!check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, nullptr, &count), err))
        return false;
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node = nullptr;
    if (!check(cudaGraphAddNode(&node, graph, deps, nullptr, count, &params), err))
        return false;
    cudaStream_t body = nullptr;
    if (!check(cudaStreamCreateWithFlags(&body, cudaStreamNonBlocking), err))
        return false;
    bool ok = check(cudaStreamBeginCaptureToGraph(body, params.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                                  cudaStreamCaptureModeThreadLocal),
                    err);
    if (ok) {
        record(body);
        cudaGraph_t ended = nullptr;
        const auto recorded = cudaGetLastError();
        const auto end = cudaStreamEndCapture(body, &ended);
        ok = check(recorded, err) && check(end, err);
    }
    cudaStreamDestroy(body);
    return ok &&
           check(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies), err);
}
#endif

bool RemoteExperts::prepare_gpu_window(GpuTaskPacket *packets, int lb, int le, int nt, const float *x, float *out,
                                       uint32_t *ready, uint32_t *abort_flag, std::string &err) {
    const OnDevice on(device_);
    if (!direct_io_ || !direct_input_ || !separate_ready_ || remote_opt_ || nt < 1 || nt > 8) {
        err = "GPU tasks require native direct input/output and separate ready";
        return false;
    }
    if (task_packets_ && task_packets_ != packets) {
        err = "helper GPU task packet address changed";
        return false;
    }
    task_packets_ = packets;
    if (task_exec_[nt])
        return true;
    auto enabled = [](const char *name) {
        const char *v = std::getenv(name);
        return v && std::atoi(v) != 0;
    };
    const bool skip_empty = enabled("STRATA_REMOTE_COLD_CONDITIONAL");
    const char *cond_env = std::getenv("STRATA_REMOTE_COLD_CONDITIONAL");
    const bool compact_cold = cond_env && std::atoi(cond_env) == 2;
    const bool early_publish = enabled("STRATA_REMOTE_EARLY_PUBLISH");
#if CUDART_VERSION < 13000
    if (skip_empty) {
        err = "conditional cold graphs require CUDA 13";
        return false;
    }
#endif
    const char *cold_env = std::getenv("STRATA_REMOTE_COLD_CAP");
    const int requested = cold_env ? std::min(16, std::max(0, std::atoi(cold_env))) : 0;
    if (requested && !cold_stream_) {
        cold_cap_ = requested;
        for (int l = lb; l < le; ++l)
            cold_stride_ = std::max<size_t>(cold_stride_, strata::kernels::cpu::expert_layout().blob_bytes(l));
        cold_stride_ = (cold_stride_ + 255) & ~size_t(255);
        if (!check(cudaMalloc(&cold_meta_, sizeof(GpuRemoteMeta)), err) ||
            !check(cudaMalloc((void **)&cold_weights_, cold_cap_ * cold_stride_), err) ||
            !check(cudaMalloc((void **)&cold_source_, cold_cap_ * sizeof(unsigned long long)), err) ||
            !check(cudaMalloc((void **)&cold_counts_, 2 * cold_cap_ * sizeof(int32_t)), err) ||
            !check(cudaStreamCreateWithFlags(&cold_stream_, cudaStreamNonBlocking), err) ||
            !check(cudaEventCreateWithFlags(&cold_fork_, cudaEventDisableTiming), err))
            return false;
        cold_done_.resize(cold_cap_, nullptr);
        for (auto &ev : cold_done_)
            if (!check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), err))
                return false;
        std::fprintf(stderr,
                     "strata helper cold: %d slots, %.1f MiB; direct VMM reads, per-expert events, captured overlap\n",
                     cold_cap_, cold_cap_ * cold_stride_ / 1048576.0);
    }
    GpuTaskPacket *mapped = nullptr;
    uint32_t *abort_mapped = nullptr;
    uint64_t aliases[3] = {};
    const void *hosts[3] = {x, out, ready};
    if (!check(cudaHostGetDevicePointer((void **)&mapped, packets, 0), err) ||
        !check(cudaHostGetDevicePointer((void **)&abort_mapped, abort_flag, 0), err))
        return false;
    for (int i = 0; i < 3; ++i)
        if (!check(cudaHostGetDevicePointer((void **)&aliases[i], const_cast<void *>(hosts[i]), 0), err))
            return false;
    for (int l = lb; l < le; ++l) {
        packets[l - lb].transfer.input = aliases[0];
        packets[l - lb].transfer.output = aliases[1];
        packets[l - lb].transfer.ready = aliases[2];
        packets[l - lb].transfer.verify = 0;
    }
    auto compute = [&](int l) {
        const auto &f = strata::kernels::cpu::expert_layout().fmt[(size_t)l];
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (f.had_gu || f.had_d) {
            err = "GPU tasks do not support Hadamard layers";
            return false;
        }
        strata::kernels::quantize_q8_1_rows(d_x_, nt, H, d_q8_, stream_);
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_, nt * 10, nt * 10, d_q8_,
                                               d_scratch_, d_out_, stream_);
        return check(cudaGetLastError(), err);
    };
    // Resolve lazy format/device setup outside capture; zero groups touch no weights.
    if (!check(cudaMemsetAsync(d_meta_, 0, sizeof(GpuRemoteMeta) + sizeof(RemoteTransferMeta), stream_), err) ||
        !check(cudaMemsetAsync(d_x_, 0, nt * H * sizeof(float), stream_), err))
        return false;
    for (int l = lb; l < le; ++l)
        if (!compute(l))
            return false;
    if (!check(cudaStreamSynchronize(stream_), err) ||
        !check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), err))
        return false;
    bool ok = true;
    for (int l = lb; l < le && ok; ++l) {
        gpu_tasks_load(mapped + l - lb, d_meta_, abort_mapped, l - lb + 1, stream_, cold_meta_);
#if CUDART_VERSION >= 13000
        ColdConditions handles{};
        cudaGraphConditionalHandle count_handle{};
        if (skip_empty && cold_cap_) {
            cudaStreamCaptureStatus status;
            cudaGraph_t parent = nullptr;
            ok = check(cudaStreamGetCaptureInfo(stream_, &status, nullptr, &parent), err);
            if (compact_cold)
                ok = ok && check(cudaGraphConditionalHandleCreate(&count_handle, parent, 0, 0), err);
            else
                for (int q = 0; q < cold_cap_ && ok; ++q)
                    ok = check(
                             cudaGraphConditionalHandleCreate(&handles.fetch[q], parent, 0, cudaGraphCondAssignDefault),
                             err) &&
                         check(cudaGraphConditionalHandleCreate(&handles.compute[q], parent, 0,
                                                                cudaGraphCondAssignDefault),
                               err);
            if (!ok)
                break;
            if (compact_cold)
                set_cold_count<<<1, 1, 0, stream_>>>((GpuRemoteMeta *)cold_meta_, count_handle);
            else
                set_cold_conditions<<<1, 32, 0, stream_>>>((GpuRemoteMeta *)cold_meta_, handles, cold_cap_);
        }
#endif
        if (cold_cap_) {
            prepare_helper_cold<<<1, 32, 0, stream_>>>((GpuRemoteMeta *)cold_meta_, cold_source_, cold_counts_,
                                                       cold_weights_, cold_stride_, cold_cap_);
            cudaEventRecord(cold_fork_, stream_);
            cudaStreamWaitEvent(cold_stream_, cold_fork_, 0);
            const size_t bytes = strata::kernels::cpu::expert_layout().blob_bytes(l);
#if CUDART_VERSION >= 13000
            if (compact_cold) {
                fetch_helper_active<<<32, 256, 0, cold_stream_>>>(cold_source_, (GpuRemoteMeta *)cold_meta_,
                                                                  cold_weights_, cold_stride_, bytes / 16,
                                                                  cold_counts_ + cold_cap_);
                cudaEventRecord(cold_done_[0], cold_stream_);
            } else
#endif
                for (int q = 0; q < cold_cap_; ++q) {
                    auto record = [&](cudaStream_t s) {
                        fetch_helper_expert<<<32, 256, 0, s>>>(cold_source_ + q, cold_counts_ + q,
                                                               (uint4 *)(cold_weights_ + q * cold_stride_), bytes / 16);
                    };
#if CUDART_VERSION >= 13000
                    if (skip_empty)
                        ok = capture_cold_if(cold_stream_, handles.fetch[q], record, err);
                    else
#endif
                        record(cold_stream_);
                    if (!ok)
                        break;
                    cudaEventRecord(cold_done_[q], cold_stream_);
                }
        }
        ok = ok && copy_input(nt, err) && compute(l);
        if (ok && early_publish)
            scatter_completed_experts<<<nt * 10, 256, 0, stream_>>>((const float4 *)d_out_, (float4 *)z_out_,
                                                                    (const RemoteTransferMeta *)d_transfer_,
                                                                    (const GpuRemoteMeta *)d_meta_, -1);
        if (ok && cold_cap_) {
            const auto &f = strata::kernels::cpu::expert_layout().fmt[(size_t)l];
            const auto layout = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            auto *meta = (GpuRemoteMeta *)cold_meta_;
#if CUDART_VERSION >= 13000
            if (compact_cold) {
                auto record = [&](int n, cudaStream_t s) {
                    for (int q = 0; q < n; ++q) {
                        wait_helper_expert<<<1, 32, 0, s>>>(cold_counts_ + cold_cap_ + q);
                        strata::kernels::native_expert_grouped(layout, meta->ptr + q, meta->start + q, cold_counts_ + q,
                                                               meta->dst, meta->tok, 1, nt * 10, d_q8_, d_scratch_,
                                                               d_out_, s);
                        if (early_publish)
                            scatter_completed_experts<<<nt * 10, 256, 0, s>>>((const float4 *)d_out_, (float4 *)z_out_,
                                                                              (const RemoteTransferMeta *)d_transfer_,
                                                                              meta, q);
                    }
                };
                ok = capture_cold_switch(stream_, count_handle, cold_cap_, record, err);
                cudaStreamWaitEvent(stream_, cold_done_[0], 0);
            } else
#endif
                for (int q = 0; q < cold_cap_; ++q) {
                    cudaStreamWaitEvent(stream_, cold_done_[q], 0);
                    auto record = [&](cudaStream_t s) {
                        strata::kernels::native_expert_grouped(layout, meta->ptr + q, meta->start + q, cold_counts_ + q,
                                                               meta->dst, meta->tok, 1, nt * 10, d_q8_, d_scratch_,
                                                               d_out_, s);
                        if (early_publish)
                            scatter_completed_experts<<<nt * 10, 256, 0, s>>>((const float4 *)d_out_, (float4 *)z_out_,
                                                                              (const RemoteTransferMeta *)d_transfer_,
                                                                              meta, q);
                    };
#if CUDART_VERSION >= 13000
                    if (skip_empty)
                        ok = capture_cold_if(stream_, handles.compute[q], record, err);
                    else
#endif
                        record(stream_);
                    if (!ok)
                        break;
                }
            ok = ok && check(cudaGetLastError(), err);
        }
        if (ok && !early_publish)
            scatter_remote_output<<<nt * 10, 256, 0, stream_>>>((const float4 *)d_out_, (float4 *)z_out_,
                                                                (const RemoteTransferMeta *)d_transfer_);
    }
    cudaGraph_t graph = nullptr;
    const auto ended = cudaStreamEndCapture(stream_, &graph);
    if (!ok || !check(ended, err)) {
        if (graph)
            cudaGraphDestroy(graph);
        return false;
    }
    const auto made = cudaGraphInstantiate(&task_exec_[nt], graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (!check(made, err) || !check(cudaGraphUpload(task_exec_[nt], stream_), err) ||
        !check(cudaStreamSynchronize(stream_), err))
        return false;
    std::fprintf(
        stderr,
        "strata gpu tasks: CUDA%d captured %d layers, T=%d; one helper launch/window; conditional=%d early=%d\n",
        device_, le - lb, nt, skip_empty, early_publish);
    return true;
}
bool RemoteExperts::launch_gpu_window(int nt, std::string &err) {
    const OnDevice on(device_);
    if (keepalive_)
        remote_keepalive_touch(keepalive_);
    ++task_launches_;
    return check(cudaGraphLaunch(task_exec_[nt], stream_), err);
}
bool RemoteExperts::finish_gpu_window(int lb, int le, std::string &err) {
    const OnDevice on(device_);
    if (!check(cudaStreamSynchronize(stream_), err))
        return false;
    for (int l = lb; l < le; ++l) {
        auto &p = task_packets_[l - lb];
        computed_ += p.transfer.entries;
        launched_layers_ += p.remote.count + p.cold.count > 0;
        returned_bytes_ += (uint64_t)p.transfer.entries * H * sizeof(float);
    }
    if (task_launches_ == 1 || task_launches_ % 512 == 0)
        std::fprintf(stderr, "strata gpu tasks: %llu helper window launches, %lld computed rows\n",
                     (unsigned long long)task_launches_, (long long)computed_);
    return true;
}
#endif // STRATA_TASK_PLANNER_TEST
} // namespace strata::core
#endif
