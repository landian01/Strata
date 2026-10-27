#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/remote_transfer.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace strata::core {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
namespace {
// One small block by default, bounded to 50 ms; never an unbounded WDDM kernel.
__global__ void keepalive_work(float* sink) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    float x = 0.5f + (float) threadIdx.x * 0.00001f;
    do {
        #pragma unroll 8
        for (int i = 0; i < 256; ++i) x = fmaf(x, 0.999999f, 0.0000003f);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < 50000000ull);
    sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
struct KeepAliveState {
    std::atomic<bool> stop{false};
    std::atomic<int64_t> last_ms{0};
    std::thread worker;
};
int64_t keepalive_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
void* remote_keepalive_start(int device, int blocks) {
    auto* state = new KeepAliveState;
    state->worker = std::thread([state, device, blocks] {
        cudaStream_t stream = nullptr;
        cudaEvent_t done = nullptr;
        float* sink = nullptr;
        int least = 0, greatest = 0;
        cudaError_t status = cudaSetDevice(device);
        if (status == cudaSuccess) status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
        if (status == cudaSuccess) status = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaMalloc((void**) &sink, blocks * 64 * sizeof(float));
        uint64_t batches = 0;
        if (status == cudaSuccess)
            std::fprintf(stderr, "strata keepalive: CUDA%d ready, %d blocks, request activity only\n", device, blocks);
        while (status == cudaSuccess && !state->stop.load(std::memory_order_relaxed)) {
            if (keepalive_ms() - state->last_ms.load(std::memory_order_relaxed) > 100) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            keepalive_work<<<blocks, 64, 0, stream>>>(sink);
            status = cudaGetLastError();
            if (status != cudaSuccess) break;
            status = cudaEventRecord(done, stream);
            if (status != cudaSuccess) break;
            do {
                status = cudaEventQuery(done);
                if (status == cudaErrorNotReady) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while (status == cudaErrorNotReady);
            ++batches;
        }
        if (status != cudaSuccess)
            std::fprintf(stderr, "strata keepalive: CUDA%d failed: %s\n", device, cudaGetErrorString(status));
        if (stream) cudaStreamSynchronize(stream);
        if (sink) cudaFree(sink);
        if (done) cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
        std::fprintf(stderr, "strata keepalive: CUDA%d stopped, %llu bounded batches\n", device, (unsigned long long) batches);
    });
    return state;
}
void remote_keepalive_touch(void* p) {
    static_cast<KeepAliveState*>(p)->last_ms.store(keepalive_ms(), std::memory_order_relaxed);
}
void remote_keepalive_stop(void* p) {
    auto* state = static_cast<KeepAliveState*>(p);
    state->stop.store(true, std::memory_order_relaxed);
    state->worker.join();
    delete state;
}
#else
void* remote_keepalive_start(int, int) { return nullptr; }
void remote_keepalive_touch(void*) {}
void remote_keepalive_stop(void*) {}
#endif
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int K = 10, MAXT = strata::kernels::cpu::MAXT, CAP = MAXT * K;
struct ReduceMeta { int32_t row[CAP]; float weight[CAP]; };

bool check(cudaError_t status, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("remote expert decode optimization: ") + cudaGetErrorString(status);
    return false;
}

// Graphs reuse a bucket of group counts. Read the current compact row count
// from uploaded metadata rather than capturing a stale host-side entry count.
__global__ void copy_remote_output(float4* __restrict__ dst, const float4* __restrict__ src,
                                    const int32_t* start, const int32_t* groups, int64_t row4) {
    const int64_t n4 = (int64_t) start[*groups] * row4;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += (int64_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

__global__ void copy_remote_input(float4* dst, const RemoteTransferMeta* meta, int64_t n4) {
    if (!meta->input || meta->entries == 0) return;
    const auto* src = (const float4*) meta->input;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += (int64_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

__global__ void scatter_remote_output(const float4* src, float4* shadow, const RemoteTransferMeta* meta) {
    const int compact = blockIdx.x;
    if (compact >= meta->entries) return;
    const int row = meta->row[compact];
    auto* dst = (float4*) meta->output;
    for (int c = threadIdx.x; c < H / 4; c += blockDim.x) {
        const float4 v = src[(int64_t) compact * (H / 4) + c];
        dst[(int64_t) row * (H / 4) + c] = v;
        if (meta->verify) shadow[(int64_t) compact * (H / 4) + c] = v;
    }
    // Every writer flushes its own mapped stores before the block publishes.
    if (meta->ready) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) ((volatile uint32_t*) meta->ready)[row] = meta->sequence;
    }
}

__global__ void reduce_experts(const float* parts, const ReduceMeta* meta, float* sum) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int i = t * K + j, r = meta->row[i];
        if (r >= 0) v += parts[(int64_t) r * H + c] * meta->weight[i];
    }
    sum[(int64_t) t * H + c] = v;
}

// Like the upstream mapped-row copy, also skip rows returned in the remote sum.
__global__ void copy_cpu_rows(float4* dst, const volatile float4* src, const int32_t* mask,
                              const int32_t* hit_rows, const int32_t* count) {
    const int row = blockIdx.x;
    __shared__ int skip;
    if (threadIdx.x == 0) {
        int s = mask[row];
        for (int i = 0; i < *count; ++i) s |= hit_rows[i] == row;
        skip = s;
    }
    __syncthreads();
    const int64_t start = (int64_t) row * (H / 4);
    for (int i = threadIdx.x; i < H / 4; i += blockDim.x)
        dst[start + i] = skip ? make_float4(0, 0, 0, 0) : const_cast<const float4*>(src)[start + i];
}
} // namespace

bool RemoteExperts::copy_output(int64_t cap_entries, std::string& err) {
    if (direct_active_) {
        scatter_remote_output<<<(int) cap_entries, 256, 0, stream_>>>(
            (const float4*) d_out_, (float4*) z_out_, (const RemoteTransferMeta*) d_transfer_);
        return check(cudaGetLastError(), err);
    }
    const int64_t n4 = cap_entries * (H / 4);
    const int blocks = (int) std::min<int64_t>(64, (n4 + 255) / 256);
    copy_remote_output<<<blocks, 256, 0, stream_>>>(
        (float4*) z_out_, (const float4*) d_out_, d_start_, d_count_, H / 4);
    return check(cudaGetLastError(), err);
}

bool RemoteExperts::copy_input(int64_t n_tok, std::string& err) {
    const int64_t n4 = n_tok * (H / 4);
    copy_remote_input<<<(int) std::min<int64_t>(64, (n4 + 255) / 256), 256, 0, stream_>>>(
        (float4*) d_x_, (const RemoteTransferMeta*) d_transfer_, n4);
    return check(cudaGetLastError(), err);
}

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
namespace {
__global__ void reset_result_ready(uint32_t* ready, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) { ((volatile uint32_t*) ready)[i] = 0; __threadfence_system(); }
}
__global__ void copy_result_ready(float4* dst, const float4* src, int width4,
                                  const int32_t* hits, const int32_t* count,
                                  const volatile uint32_t* ready, uint32_t sequence,
                                  const volatile uint32_t* abort_flag) {
    const int row = blockIdx.x;
    __shared__ int hit;
    if (threadIdx.x == 0) {
        int h = 0;
        for (int i = 0; i < *count; ++i) h |= hits[i] == row;
        hit = h;
        if (!h) {
            while (ready[row] < sequence && *abort_flag != UINT32_MAX) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
            __threadfence_system();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width4; i += blockDim.x)
        dst[(int64_t) row * width4 + i] = hit ? make_float4(0,0,0,0) : src[(int64_t) row * width4 + i];
}
}
void remote_reset_ready(uint32_t* ready, int rows, void* stream) {
    reset_result_ready<<<(rows + 127) / 128,128,0,(cudaStream_t) stream>>>(ready, rows);
}
void remote_copy_ready(float* dst, const float* src, int rows, int width,
                       const int32_t* hits, const int32_t* count, const uint32_t* ready,
                       uint32_t sequence, const uint32_t* abort_flag, void* stream) {
    copy_result_ready<<<rows,256,0,(cudaStream_t) stream>>>((float4*) dst, (const float4*) src, width/4,
                                                         hits,count,ready,sequence,abort_flag);
}
#endif

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

bool RemoteExpertOpt::init(std::string& err) {
    void* mapped = nullptr;
    if (!check(cudaHostAlloc((void**) &h_sum_, MAXT * (H + K) * sizeof(float),
                             cudaHostAllocPortable | cudaHostAllocMapped), err) ||
        !check(cudaHostGetDevicePointer(&mapped, h_sum_, 0), err)) return false;
    m_sum_ = (float*) mapped;
    h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + MAXT * H);
    m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + MAXT * H);
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (!check(cudaMalloc((void**) &p.sum, MAXT * H * sizeof(float)), err)) return false;
    }
    return true;
}

RemoteExpertOpt::~RemoteExpertOpt() {
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (p.remote->stream_) cudaStreamSynchronize(p.remote->stream_);
        if (p.sum) cudaFree(p.sum);
        p.remote->remote_opt_ = nullptr;
    }
    if (h_sum_) cudaFreeHost(h_sum_);
}

size_t RemoteExpertOpt::metadata_bytes() { return sizeof(ReduceMeta); }

void RemoteExpertOpt::begin(const float* weights, int token_begin, int tokens) {
    weights_ = weights;
    token_begin_ = token_begin;
    tokens_ = tokens;
    std::memset(h_sum_ + token_begin * H, 0, tokens * H * sizeof(float));
    std::memset(h_mask_ + token_begin * K, 0, tokens * K * sizeof(int32_t));
}

void RemoteExpertOpt::prepare(const RemoteExperts& remote, void* metadata) const {
    auto* m = (ReduceMeta*) metadata;
    std::fill(m->row, m->row + tokens_ * K, -1);
    for (size_t i = 0; i < remote.original_row_.size(); ++i)
        m->row[remote.original_row_[i]] = (int32_t) i;
    std::memcpy(m->weight, weights_, tokens_ * K * sizeof(float));
}

bool RemoteExpertOpt::reduce(RemoteExperts& remote, const void* metadata, std::string& err) {
    const auto& p = *std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    // finish() synchronizes this stream before the host reads h_out_. When mapped output is available,
    // write the reduced rows there directly, avoiding a separate device-to-host copy for every layer.
    float* sum = remote.zero_copy_ ? remote.z_out_ : p.sum;
    reduce_experts<<<dim3((H + 255) / 256, tokens_), 256, 0, remote.stream_>>>(
        remote.d_out_, (const ReduceMeta*) metadata, sum);
    if (remote.zero_copy_) return check(cudaGetLastError(), err);
    return check(cudaMemcpyAsync(remote.h_out_, p.sum, tokens_ * H * sizeof(float),
                                  cudaMemcpyDeviceToHost, remote.stream_), err);
}

void RemoteExpertOpt::accumulate(const RemoteExperts& remote) {
    for (int64_t i = 0; i < tokens_ * H; ++i) h_sum_[token_begin_ * H + i] += remote.h_out_[i];
    for (int i = 0; i < tokens_ * K; ++i)
        h_mask_[token_begin_ * K + i] |= remote.owned_[(size_t) i] != 0;
}

void RemoteExpertOpt::copy_rows(float* dst, const float* src, int token_begin, int tokens,
                         const int32_t* hit_rows, const int32_t* count, void* stream) const {
    copy_cpu_rows<<<tokens * K, 128, 0, (cudaStream_t) stream>>>(
        (float4*) dst, (const volatile float4*) src, m_mask_ + token_begin * K, hit_rows, count);
}

void RemoteExpertOpt::combine(float* dst, float* scratch, int token_begin, int tokens,
                       const uint32_t* skip, uint32_t ring, void* stream) const {
    const float* sum = m_sum_ + token_begin * H;
    if (skip) {
        // An all-primary device-planned group bypasses the host: ignore its stale sum.
        strata::kernels::copy_or_zero_from_mapped(scratch, sum, tokens * H, skip, ring, stream);
        sum = scratch;
    }
    strata::kernels::add_inplace(dst, sum, tokens * H, stream);
}

bool RemoteExpertOpt::owns(int64_t layer, int32_t expert) const {
    for (const auto& p : peers_)
        if (p.remote->cache_.slot_of(layer, expert) >= 0) return true;
    return false;
}

bool RemoteExpertOpt::adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                     const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
                     ExpertSource& source) {
    if (max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> resident(primary.size(), 0);
    for (size_t i = 0; i < primary.size(); ++i) resident[i] = primary[i] >= 0;
    for (const auto& [i, slot] : pending) resident[(size_t) i] = 1;
    for (const auto& p : peers_) {
        const auto& r = *p.remote;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l)
            for (int32_t e = 0; e < r.n_expert_; ++e)
                if (r.cache_.slot_of(l, e) >= 0) resident[(size_t) l * r.n_expert_ + e] = 1;
    }
    struct Swap { float gain; int32_t layer, in, out; };
    std::string err;
    for (auto& p : peers_) {
        auto& r = *p.remote;
        std::vector<Swap> swaps;
        std::vector<std::pair<float, int32_t>> candidates, victims;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l) {
            candidates.clear(); victims.clear();
            const size_t base = (size_t) l * r.n_expert_;
            float primary_cold = 1.0e30f;
            for (int32_t e = 0; e < r.n_expert_; ++e) {
                const float u = usage[base + e];
                if (primary[base + e] >= 0) primary_cold = std::min(primary_cold, u);
                if (r.cache_.slot_of(l, e) >= 0) victims.emplace_back(u, e);
                else if (!resident[base + e] && u >= 2.0f) candidates.emplace_back(u, e);
            }
            const size_t n = std::min(candidates.size(), victims.size());
            if (!n) continue;
            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::partial_sort(victims.begin(), victims.begin() + n, victims.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < n; ++i) {
                if (victims[i].first > primary_cold) break;
                if (candidates[i].first < victims[i].first + 1.5f) break;
                swaps.push_back({candidates[i].first - victims[i].first, l, candidates[i].second, victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
        if (swaps.empty()) continue;
        const OnDevice on(r.device_);
        for (const auto& s : swaps) {
            const uint8_t* in_blob = source.blob(s.layer, s.in);
            if (!r.cache_.fill_slot(r.cache_.slot_of(s.layer, s.out), in_blob,
                                    r.stream_, err, (int64_t) lay.blob_bytes(s.layer))) return false;
            source.note_async_read(in_blob, r.stream_);   // #1237
        }
        if (!check(cudaStreamSynchronize(r.stream_), err)) return false;
        for (const auto& s : swaps) {
            r.cache_.replace(s.layer, s.out, s.in);
            const size_t base = (size_t) s.layer * r.n_expert_;
            resident[base + s.out] = 0;
            resident[base + s.in] = 1;
        }
    }
    return true;
}
} // namespace strata::core

#include "gpu_tasks.cuh"
