#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_transfer.hpp"
#include "strata/core/gpu_tasks.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <set>
#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL) && CUDART_VERSION >= 13000
#include <cuda.h>
#define STRATA_COLD_HOST_VMM 1
#endif

namespace strata::core {
void* remote_keepalive_start(int device, int blocks);
void remote_keepalive_touch(void* state);
void remote_keepalive_stop(void* state);
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
using RemoteMeta = GpuRemoteMeta;

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}
} // namespace

// Graphs reference this helper's fixed buffers; close() drains the stream and
// destroys them before releasing any buffers or cached expert weights.
struct RemoteGraphCache {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    struct Entry { cudaGraphExec_t exec = nullptr; bool checked = false; };
    std::unordered_map<uint64_t, Entry> entries;
    std::vector<float> reference;
    bool enabled = true, verify = false;
    uint64_t captures = 0, replays = 0, warmups = 0, fallbacks = 0, checks = 0;
    RemoteGraphCache() {
        entries.reserve(512);
        const char* v = std::getenv("STRATA_REMOTE_GRAPH_VERIFY");
        verify = v && std::atoi(v) != 0;
        if (verify) reference.resize((size_t) CAP * H);
    }
    ~RemoteGraphCache() {
        for (auto& item : entries) if (item.second.exec) cudaGraphExecDestroy(item.second.exec);
    }
#endif
};

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    if (device == primary_device()) {   // the primary may be any visible ordinal; the helper cannot be it
        err = "CUDA" + std::to_string(device) + " experts: the helper cannot be the primary device "
                                                 "(STRATA_PRIMARY_DEVICE)";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    if (keepalive_) { remote_keepalive_stop(keepalive_); keepalive_ = nullptr; }
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        if (read_stream_) cudaStreamSynchronize(read_stream_);
        if (graphs_) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
            std::fprintf(stderr, "strata remote graph: CUDA%d captures %llu, replays %llu, warmups %llu, fallbacks %llu, bitwise checks %llu\n",
                         device_, (unsigned long long) graphs_->captures, (unsigned long long) graphs_->replays,
                         (unsigned long long) graphs_->warmups, (unsigned long long) graphs_->fallbacks,
                         (unsigned long long) graphs_->checks);
#endif
            delete graphs_; graphs_ = nullptr;
        }
        if (group_buffer_) std::fprintf(stderr, "strata remote groups: CUDA%d linear buffered grouping, reference checks %llu\n",
                                       device_, (unsigned long long) group_checks_);
        if (output_kernel_) std::fprintf(stderr, "strata remote output: CUDA%d kernel writeback, calls %llu, bitwise checks %llu\n",
                                         device_, (unsigned long long) output_calls_, (unsigned long long) output_checks_);
        if (direct_io_) std::fprintf(stderr, "strata remote direct: CUDA%d %llu direct result batches, shared input %d\n",
                                     device_, (unsigned long long) direct_calls_, direct_input_);
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        for (auto& graph : task_exec_) { if (graph) cudaGraphExecDestroy(graph); graph = nullptr; }
        task_packets_ = nullptr;
        if (cold_stream_) { cudaStreamSynchronize(cold_stream_); cudaStreamDestroy(cold_stream_); cold_stream_=nullptr; }
        if (cold_fork_) { cudaEventDestroy(cold_fork_); cold_fork_=nullptr; }
        for (auto ev : cold_done_) if (ev) cudaEventDestroy(ev);
        cold_done_.clear();
        if (cold_meta_) { cudaFree(cold_meta_); cold_meta_=nullptr; }
        if (cold_weights_) { cudaFree(cold_weights_); cold_weights_=nullptr; }
        if (cold_source_) { cudaFree(cold_source_); cold_source_=nullptr; }
        if (cold_counts_) { cudaFree(cold_counts_); cold_counts_=nullptr; }
        cold_cap_=0; cold_stride_=0;
#endif
#ifdef STRATA_COLD_HOST_VMM
        CUmemAccessDesc revoke{};revoke.location.type=CU_MEM_LOCATION_TYPE_DEVICE;revoke.location.id=device_;revoke.flags=CU_MEM_ACCESS_FLAGS_PROT_NONE;
        for(const auto& range:cold_ranges_)cuMemSetAccess((CUdeviceptr)range.first,(size_t)range.second,&revoke,1);
#endif
        cold_ranges_.clear();cold_map_.clear();cold_map_base_=cold_map_chunk_=0;
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scales_) cudaFree(d_scales_);
        if (d_xr_) cudaFree(d_xr_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_meta_) cudaFree(d_meta_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_base_ || h_meta_) cudaFreeHost(h_meta_base_ ? h_meta_base_ : h_meta_);
        if (stream_) cudaStreamDestroy(stream_);
        if (read_stream_) cudaStreamDestroy(read_stream_);
    }
    device_ = -1;
    stream_ = nullptr;
    read_stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = h_meta_base_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_xr_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
    group_buffer_ = group_verify_ = false;
    group_epoch_ = 0;
    group_checks_ = 0;
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size) {
    close();
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts ||
        device == primary_device()) {   // the helper cannot be the primary device (any visible ordinal)
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
            if (auto_size && needed + bytes + (512ull << 20) > free_bytes) break;
            selected.push_back(pair);
            needed += bytes;
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                                          : " experts: no unclaimed experts remain");
        close(); return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    direct_io_ = direct_input_ = direct_active_ = false;
    direct_calls_ = 0;
    mapped_aliases_.clear();
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    const char* dio = std::getenv("STRATA_REMOTE_DIRECT_IO");
    direct_io_ = dio && std::atoi(dio) != 0 && lay.native && !remote_opt_;
    const char* din = std::getenv("STRATA_REMOTE_DIRECT_INPUT");
    direct_input_ = direct_io_ && din && std::atoi(din) != 0;
#endif
    const char* sep = std::getenv("STRATA_REMOTE_SEPARATE_READY");
    separate_ready_ = sep && std::atoi(sep) != 0;
    if (separate_ready_ && (!direct_io_ || !direct_input_)) {
        err = "separate helper completion requires direct IO and shared input"; return false;
    }
    meta_slot_ = 0;
    static_assert(CAP == 80);
    const size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ ? remote_opt_->metadata_bytes() :
                                                 direct_io_ ? sizeof(RemoteTransferMeta) : 0);
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes * (separate_ready_ ? 2 : 1), cudaHostAllocPortable), "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err, device) &&
        (!strata::kernels::cpu::hadamard_spec().any() ||
         check(cudaMalloc((void**) &d_xr_, (size_t) CAP * H * sizeof(float)), "rotated input", err, device));
    if (!allocated) { close(); return false; }
    if (!check(cudaStreamCreateWithFlags(&read_stream_, cudaStreamNonBlocking), "cache read stream", err, device)) {
        close();
        return false;
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) FF), (int) FF);
    }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    if (direct_io_ && !zero_copy_) { err = "remote direct IO requires mapped pinned buffers"; return false; }
    d_transfer_ = direct_io_ ? (void*) ((RemoteMeta*) d_meta_ + 1) : nullptr;
    output_kernel_ = output_verify_ = false;
    output_calls_ = output_checks_ = 0;
    output_reference_.clear();
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    const char* ok = std::getenv("STRATA_REMOTE_OUTPUT_KERNEL");
    output_kernel_ = (direct_io_ || (ok && std::atoi(ok) != 0)) && zero_copy_ && lay.native && !remote_opt_;
    const char* ov = std::getenv("STRATA_REMOTE_OUTPUT_VERIFY");
    output_verify_ = output_kernel_ && ov && std::atoi(ov) != 0;
    if (output_verify_) output_reference_.resize((size_t) CAP * H);
    if (output_kernel_)
        std::fprintf(stderr, "strata remote output: CUDA%d device staging plus mapped writeback enabled, direct IO %d, shared input %d\n", device_, direct_io_, direct_input_);
#endif
    h_meta_base_ = h_meta_;
    meta_stride_ = meta_bytes;
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    const char* gb = std::getenv("STRATA_REMOTE_GROUP_BUFFER");
    group_buffer_ = gb && std::atoi(gb) != 0;
    if (group_buffer_) {
        expert_epoch_.assign((size_t) experts, 0);
        expert_group_.resize((size_t) experts);
        group_size_.resize(CAP);
        group_cursor_.resize(CAP);
        const char* gv = std::getenv("STRATA_REMOTE_GROUP_VERIFY");
        group_verify_ = gv && std::atoi(gv) != 0;
        std::fprintf(stderr, "strata remote groups: CUDA%d preallocated linear grouping enabled\n", device_);
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    const char* graph = std::getenv("STRATA_REMOTE_GRAPH");
    if (graph && std::atoi(graph) != 0 && lay.native && zero_copy_ && !remote_opt_) {
        graphs_ = new RemoteGraphCache;
        std::fprintf(stderr, "strata remote graph: CUDA%d shape cache enabled, limit 512\n", device_);
    }
#endif
    const char* cold_cap_env=std::getenv("STRATA_REMOTE_COLD_CAP");
    if(cold_cap_env && std::atoi(cold_cap_env)>0) {
#ifdef STRATA_COLD_HOST_VMM
        const char* arena_vmm=std::getenv("STRATA_ARENA_VMM");
        if(!arena_vmm || std::atoi(arena_vmm)!=1) {err="helper cold requires a host VMM arena";return false;}
        const char* map_gib_env=std::getenv("STRATA_REMOTE_COLD_MAP_GIB");
        const int map_gib=map_gib_env?std::clamp(std::atoi(map_gib_env),1,16):12;
        CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_HOST;
        size_t gran=0;CUresult result=cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if(result!=CUDA_SUCCESS || !gran){err="cannot query host VMM access granularity";return false;}
        const char* chunk_env=std::getenv("STRATA_ARENA_VMM_CHUNK_MIB");
        const uint64_t chunk_target=chunk_env ? uint64_t(std::clamp(std::atoi(chunk_env),2,1024))<<20 : 1ull<<30;
        const uint64_t mapping_chunk=std::max<uint64_t>(gran,(chunk_target/gran)*gran);
        const uint64_t arena_base=(uint64_t)source.blob_stable(0,0);
        cold_map_base_=arena_base; cold_map_chunk_=mapping_chunk;
        const uint64_t end_weights=(uint64_t)source.blob_stable(layers-1,experts-1)+lay.blob_bytes(layers-1);
        const uint64_t arena_end=arena_base+((end_weights-arena_base+lay.max_blob+gran-1)/gran)*gran;
        // Windows host VMM requires permission changes to cover each full
        // cuMemMap allocation, even when allocation granularity is smaller.
        auto bounds=[&](uint64_t host,uint64_t bytes) {
            const uint64_t begin=arena_base+(host-arena_base)/mapping_chunk*mapping_chunk;
            const uint64_t end=arena_base+((host-arena_base+bytes+mapping_chunk-1)/mapping_chunk)*mapping_chunk;
            return std::make_pair(begin,end);
        };
        std::set<uint64_t> pages;
        cold_map_.assign((size_t)layers*experts,0);
        for(const auto& pair:ranked) {
            const int l=pair.first,e=pair.second;
            if(primary.slot_of(l,e)>=0 || cache_.slot_of(l,e)>=0)continue;
            const auto* host=source.blob_stable(l,e);if(!host)continue;
            const auto range=bounds((uint64_t)host,lay.blob_bytes(l));
            size_t extra=0;for(uint64_t page=range.first;page<range.second;page+=mapping_chunk)extra+=pages.count(page)==0;
            if((pages.size()+extra)*mapping_chunk>(uint64_t(map_gib)<<30))continue;
            for(uint64_t page=range.first;page<range.second;page+=mapping_chunk)pages.insert(page);
        }
        for(uint64_t page:pages)cold_ranges_.emplace_back(page,std::min<uint64_t>(mapping_chunk,arena_end-page));
        // Neighbouring experts covered by the same allocations are usable too.
        for(int l=0;l<layers;++l)for(int e=0;e<experts;++e) {
            const uint64_t host=(uint64_t)source.blob_stable(l,e);if(!host)continue;
            const auto range=bounds(host,lay.blob_bytes(l));bool covered=true;
            for(uint64_t page=range.first;page<range.second;page+=mapping_chunk)covered=covered && pages.count(page)>0;
            if(covered)cold_map_[(size_t)l*experts+e]=host;
        }
        CUmemAccessDesc access{};access.location.type=CU_MEM_LOCATION_TYPE_DEVICE;access.location.id=device_;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READ;
        for(const auto& range:cold_ranges_) {
            result=cuMemSetAccess((CUdeviceptr)range.first,(size_t)range.second,&access,1);
            if(result!=CUDA_SUCCESS) {const char* name=nullptr;cuGetErrorName(result,&name);err=std::string("bounded helper VMM access: ")+(name?name:"driver error");return false;}
        }
        size_t free=0,total=0;
        if(!check(cudaMemGetInfo(&free,&total),"bounded helper mapping VRAM query",err,device_))return false;
        std::fprintf(stderr,"strata helper cold mapping: %.3f GiB, %zu ranges, %zu expert blobs; %.1f MiB VRAM free; unmapped misses stay primary\n",
            pages.size()*mapping_chunk/1073741824.,cold_ranges_.size(),std::count_if(cold_map_.begin(),cold_map_.end(),[](uint64_t x){return x!=0;}),free/1048576.);
#else
        err="bounded helper host VMM needs Windows CUDA 13";return false;
#endif
    }
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    cache_upload_bytes_ = cache_swap_count_ = cache_rebalances_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    if (const char* ka = std::getenv("STRATA_REMOTE_KEEPALIVE"))
        if (std::atoi(ka) > 0) keepalive_ = remote_keepalive_start(device_, std::clamp(std::atoi(ka), 1, 4));
    return true;
}

bool RemoteExperts::rebalance_cache(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                                    const std::vector<std::pair<int32_t,int32_t>>& pending, int max_swaps,
                                    ExpertSource& source, std::string& err) {
    const size_t total = layers_present_.size() * (size_t)n_expert_;
    if (max_swaps <= 0) return true;
    if (device_ < 0 || remote_opt_ || usage.size() != total || primary.size() != total) {
        err = "helper cache rebalance: incompatible state"; return false;
    }
    const auto& layout = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> reserved(total, 0);
    for (size_t i = 0; i < total; ++i) reserved[i] = primary[i] >= 0;
    for (const auto& pair : pending) {
        if (pair.first < 0 || (size_t)pair.first >= total) { err = "helper cache rebalance: invalid pending expert"; return false; }
        reserved[(size_t)pair.first] = 1;
    }
    struct Swap { double saved_bytes; int layer, incoming, outgoing, slot; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float,int>> candidates, victims;
    for (int l = 0; l < (int)layers_present_.size(); ++l) {
        candidates.clear(); victims.clear();
        const size_t base = (size_t)l * n_expert_;
        for (int e = 0; e < n_expert_; ++e) {
            if (cache_.slot_of(l,e) >= 0) {
                if (reserved[base+e]) { err = "helper cache rebalance: overlapping ownership"; return false; }
                victims.emplace_back(usage[base+e],e);
            }
            else if (!reserved[base+e] && usage[base+e] >= 2.0f) candidates.emplace_back(usage[base+e],e);
        }
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.first>b.first || (a.first==b.first && a.second<b.second);});
        std::sort(victims.begin(),victims.end());
        for (size_t j = 0; j < std::min(candidates.size(),victims.size()); ++j) {
            const float gain = candidates[j].first-victims[j].first;
            if (gain < 1.5f) break;
            swaps.push_back({(double)gain*layout.blob_bytes(l),l,candidates[j].second,victims[j].second,
                             cache_.slot_of(l,victims[j].second)});
        }
    }
    std::stable_sort(swaps.begin(),swaps.end(),[](const auto& a,const auto& b){return a.saved_bytes>b.saved_bytes;});
    if ((int)swaps.size()>max_swaps) swaps.resize(max_swaps);
    if (swaps.empty()) return true;
    DeviceScope on(device_);
    if (!on.ok) { err=on.error(device_); return false; }
    // A caller accidentally entering during a window must fail, not overwrite
    // weights still referenced by the pre-recorded helper execution graph.
    if (!check(cudaStreamQuery(stream_),"helper rebalance requires an idle stream",err,device_)) return false;
    uint64_t uploaded=0;
    for (const auto& swap : swaps) {
        const uint8_t* blob=source.blob_stable(swap.layer,swap.incoming);
        if (!blob) { err = "helper cache rebalance: source expert unavailable"; return false; }
        if (cold_map_.empty()) {
            if (!cache_.fill_slot(swap.slot,blob,stream_,err,(int64_t)layout.blob_bytes(swap.layer))) return false;
        } else {
            // Host VMM H2D cannot straddle allocations with different helper
            // access permissions. Split DMA at the same map boundaries; do
            // not copy the weight bytes through an extra CPU staging buffer.
            const uint64_t chunk=cold_map_chunk_, base=cold_map_base_;
            uint8_t* target=cache_.device_slot(swap.slot);
            const size_t bytes=(size_t)layout.blob_bytes(swap.layer);
            for(size_t offset=0;offset<bytes;) {
                const size_t part=std::min<size_t>(bytes-offset,chunk-(((uint64_t)blob+offset-base)%chunk));
                if(!check(cudaMemcpyAsync(target+offset,blob+offset,part,cudaMemcpyHostToDevice,stream_),
                          "helper cache refill DMA segment",err,device_))return false;
                offset+=part;
            }
        }
        source.note_async_read(blob,stream_);
        uploaded+=layout.blob_bytes(swap.layer);
    }
    if (!check(cudaStreamSynchronize(stream_),"helper cache refill",err,device_)) return false;
    static const bool verify=[] {const char* v=std::getenv("STRATA_REMOTE_CACHE_VERIFY");return v && std::atoi(v)!=0;}();
    for (const auto& swap : swaps) {
        if (verify && !cache_.verify_slot(swap.slot,source.blob_stable(swap.layer,swap.incoming),err,
                                         (int64_t)layout.blob_bytes(swap.layer))) return false;
        cache_.replace(swap.layer,swap.outgoing,swap.incoming);
    }
    cache_upload_bytes_+=uploaded;cache_swap_count_+=swaps.size();++cache_rebalances_;
    if (cache_rebalances_==1 || cache_rebalances_%32==0)
        std::fprintf(stderr,"strata helper cache: %llu drained-boundary rounds, %llu swaps, %llu uploaded bytes; verify %d\n",
                     (unsigned long long)cache_rebalances_,(unsigned long long)cache_swap_count_,
                     (unsigned long long)cache_upload_bytes_,verify);
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err, float* out, uint32_t* ready, uint32_t sequence, const RoutePlan* plan) {
    if (keepalive_) remote_keepalive_touch(keepalive_);
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    direct_active_ = direct_io_ && out != nullptr;
    async_active_ = separate_ready_ && direct_active_ && ready != nullptr && sequence != 0;
    if (separate_ready_) {
        // At most two routed groups are in flight. The primary consumes each
        // group's prior result (and hence its metadata) before reusing it.
        meta_slot_ ^= 1;
        h_meta_ = (char*) h_meta_base_ + meta_slot_ * meta_stride_;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    if (plan) {
        if (plan->groups < 0 || plan->groups > CAP || plan->entries < 0 || plan->entries > n ||
            plan->start[0] != 0 || plan->start[plan->groups] != plan->entries) {
            err = "invalid unified remote plan"; return false;
        }
        for (int g = 0; g < plan->groups; ++g) {
            if (plan->start[g] >= plan->start[g+1] || plan->slot[g] < 0) {
                err = "invalid unified remote group"; return false;
            }
            group_id_.push_back(plan->expert[g]);
            ptr_.push_back((unsigned long long) cache_.device_slot(plan->slot[g]));
            start_.push_back(plan->start[g]);
        }
        start_.push_back(plan->entries);
        for (int j = 0; j < plan->entries; ++j) {
            const int i = plan->row[j];
            if (i < 0 || i >= n || owned_[(size_t) i]) {
                err = "invalid or duplicate unified remote row"; return false;
            }
            owned_[(size_t) i] = 1;
            original_row_.push_back(i); dst_.push_back(j); tok_.push_back((int32_t)(i/k));
        }
        computed_ += plan->entries;
        if (group_verify_) {
            for (int i = 0; i < n; ++i) {
                const int e = ids[i];
                const bool expected = (!kind || kind[i] == -1) && e >= 0 && e < n_expert_ &&
                    (!primary_res || primary_res[(size_t)layer * n_expert_ + e] < 0) && holds(layer,e);
                if (expected != (owned_[(size_t)i] != 0)) { err = "unified remote ownership differs"; return false; }
            }
            int row = 0;
            for (int g = 0; g < plan->groups; ++g) {
                if (cache_.slot_of(layer,plan->expert[g]) != plan->slot[g] || plan->start[g] != row) {
                    err = "unified remote slot or group differs"; return false;
                }
                for (int i = 0; i < n; ++i) if (owned_[(size_t)i] && ids[i] == plan->expert[g]) {
                    if (row >= plan->entries || plan->row[row++] != i) {
                        err = "unified remote row order differs"; return false;
                    }
                }
            }
            if (row != plan->entries) { err = "unified remote count differs"; return false; }
            ++group_checks_;
        }
        if (group_id_.empty()) return true;
    } else {
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    if (group_buffer_ && ++group_epoch_ == 0) {
        std::fill(expert_epoch_.begin(), expert_epoch_.end(), 0);
        group_epoch_ = 1;
    }
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        if (group_buffer_) {
            if (expert_epoch_[(size_t) e] == group_epoch_) group = expert_group_[(size_t) e];
        } else {
            for (size_t g = 0; g < group_id_.size(); ++g)
                if (group_id_[g] == e) { group = (int32_t) g; break; }
        }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
            if (group_buffer_) {
                expert_epoch_[(size_t) e] = group_epoch_;
                expert_group_[(size_t) e] = group;
                group_size_[(size_t) group] = 0;
            }
        }
        group_of_[(size_t) i] = group;
        if (group_buffer_) ++group_size_[(size_t) group];
        ++computed_;
    }
    if (group_id_.empty()) return true;
    if (group_buffer_) {
        start_.resize(group_id_.size() + 1);
        start_[0] = 0;
        for (size_t g = 0; g < group_id_.size(); ++g) {
            start_[g + 1] = start_[g] + group_size_[g];
            group_cursor_[g] = start_[g];
        }
        const size_t entries = (size_t) start_.back();
        original_row_.resize(entries); dst_.resize(entries); tok_.resize(entries);
        // Stable counting scatter: same first-seen expert and per-expert token
        // order as the old nested scans, with no allocations after open().
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] >= 0) {
            const size_t row = (size_t) group_cursor_[(size_t) group_of_[(size_t) i]]++;
            original_row_[row] = (int32_t) i;
            dst_[row] = (int32_t) row;
            tok_[row] = (int32_t) (i / k);
        }
        if (group_verify_) {
            size_t row = 0;
            bool same = true;
            for (size_t g = 0; g < group_id_.size(); ++g) {
                same = same && start_[g] == (int32_t) row;
                for (int64_t i = 0; i < n; ++i) if (owned_[(size_t) i] && ids[i] == group_id_[g]) {
                    same = same && row < entries && group_of_[(size_t) i] == (int32_t) g &&
                           original_row_[row] == i && dst_[row] == (int32_t) row && tok_[row] == i / k;
                    ++row;
                }
            }
            if (!same || row != entries || start_.back() != (int32_t) row) {
                err = "remote buffered grouping differs from stable reference";
                return false;
            }
            ++group_checks_;
        }
    } else {
        for (size_t g = 0; g < group_id_.size(); ++g) {
            start_.push_back((int32_t) dst_.size());
            for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
                // Dense return rows; the kernels select inputs through tok_.
                original_row_.push_back((int32_t) i);
                dst_.push_back((int32_t) dst_.size());
                tok_.push_back((int32_t) (i / k));
            }
        }
        start_.push_back((int32_t) dst_.size());
    }
    } // legacy classification/grouping
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    if (!direct_active_ || !direct_input_ || output_verify_)
        std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active();
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    if (direct_active_) {
        auto alias = [&](const void* p, uint64_t& result) {
            for (const auto& item : mapped_aliases_) if (item.first == p) {
                result = (uint64_t) item.second; return true;
            }
            void* mapped = nullptr;
            if (!check(cudaHostGetDevicePointer(&mapped, const_cast<void*>(p), 0), "map shared IO", err, device_)) return false;
            mapped_aliases_.push_back({p, mapped}); result = (uint64_t) mapped; return true;
        };
        auto* transfer = (RemoteTransferMeta*) (meta + 1);
        *transfer = {};
        if (!alias(out, transfer->output) || (direct_input_ && !alias(x, transfer->input))) return false;
        transfer->entries = (int32_t) original_row_.size();
        transfer->verify = output_verify_;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        transfer->verify = transfer->verify || (graphs_ && graphs_->verify);
#endif
        if (async_active_) {
            if (!alias(ready, transfer->ready)) return false;
            transfer->sequence = sequence;
            // Diagnostic replays must finish before the consumer can read.
            // Otherwise a reference pass would rewrite a live output buffer.
            if (transfer->verify) transfer->ready = 0;
        }
        std::copy(original_row_.begin(), original_row_.end(), transfer->row);
        ++direct_calls_;
    }
    groups_ = (int32_t) group_id_.size();
    bool launched = false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    if (graphs_ && graphs_->enabled) {
        auto& gc = *graphs_;
        const auto& fmt = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
        // A graph reads fresh pointers/indices from d_meta_. Its scalar layout
        // is reusable across layers with the same format and geometry.
        if (!fmt.had_gu && !fmt.had_d && fmt.n_embd == H && fmt.n_ff == FF) {
            int cap_groups = 1;
            // Preserve the SM75 down-layout boundary at 30 groups. Ordinary
            // power-of-two rounding could otherwise change the chosen kernel.
            for (int cap : {1, 2, 4, 8, 16, 30, 40, (int) CAP}) {
                cap_groups = cap;
                if (groups_ <= cap) break;
            }
            const uint64_t key = ((uint64_t) fmt.gu_type << 48) | ((uint64_t) fmt.d_type << 32) |
                                 ((uint64_t) n_tok << 16) | (uint64_t) cap_groups | ((uint64_t) direct_active_ << 63) | ((uint64_t) meta_slot_ << 62);
            auto it = gc.entries.find(key);
            if (it == gc.entries.end() && gc.entries.size() < 512) {
                // First run outside capture initializes any lazy CUDA tables.
                gc.entries.emplace(key, RemoteGraphCache::Entry{});
                ++gc.warmups;
                if (!enqueue(layer, n_tok, cap_groups, n, err)) return false;
                launched = true;
            } else if (it != gc.entries.end()) {
                auto& entry = it->second;
                if (!entry.exec) {
                    cudaGraph_t graph = nullptr;
                    cudaError_t status = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
                    if (status == cudaSuccess) {
                        const bool queued = enqueue(layer, n_tok, cap_groups, n, err);
                        status = cudaStreamEndCapture(stream_, &graph);
                        if (status == cudaSuccess && !queued) status = cudaErrorStreamCaptureInvalidated;
                        if (status == cudaSuccess)
                            status = cudaGraphInstantiate(&entry.exec, graph, nullptr, nullptr, 0);
                    }
                    if (graph) cudaGraphDestroy(graph);
                    if (status != cudaSuccess) {
                        std::fprintf(stderr, "strata remote graph: CUDA%d disabled after capture error: %s\n",
                                     device_, cudaGetErrorString(status));
                        gc.enabled = false;
                        err.clear();
                        cudaGetLastError();
                    } else ++gc.captures;
                }
                if (entry.exec && gc.enabled) {
                    if (!check(cudaGraphLaunch(entry.exec, stream_), "graph launch", err, device_)) return false;
                    ++gc.replays;
                    launched = true;
                    // Diagnostic only: compare real inputs and changing routes
                    // against the original, unbucketed eager call. Never timed
                    // as a performance run when this check is enabled.
                    if (gc.verify && (!entry.checked || gc.replays % 97 == 0)) {
                        if (!check(cudaStreamSynchronize(stream_), "graph verify sync", err, device_)) return false;
                        const size_t bytes = dst_.size() * H * sizeof(float);
                        std::memcpy(gc.reference.data(), h_out_, bytes);
                        if (!enqueue(layer, n_tok, groups_, (int64_t) dst_.size(), err) ||
                            !check(cudaStreamSynchronize(stream_), "eager verify sync", err, device_)) return false;
                        if (std::memcmp(gc.reference.data(), h_out_, bytes) != 0) {
                            err = "remote CUDA Graph differs bitwise from eager reference at layer " + std::to_string(layer);
                            return false;
                        }
                        entry.checked = true;
                        ++gc.checks;
                    }
                }
            }
        }
        if (!launched) ++gc.fallbacks;
    }
#endif
    if (!launched && !enqueue(layer, n_tok, groups_, (int64_t) dst_.size(), err)) return false;
    if (output_kernel_) {
        ++output_calls_;
        if (output_verify_ && (output_calls_ == 1 || output_calls_ % 97 == 0)) {
            if (!check(cudaStreamSynchronize(stream_), "output verify sync", err, device_)) return false;
            const size_t bytes = dst_.size() * H * sizeof(float);
            std::memcpy(output_reference_.data(), h_out_, bytes);
            if (direct_active_) for (size_t i = 0; i < original_row_.size(); ++i)
                if (std::memcmp(out + (size_t) original_row_[i] * H, h_out_ + i * H, H * sizeof(float)) != 0) {
                    err = "remote direct scatter differs from compact rows"; return false;
                }
            // Same inputs and routes, with the previous direct mapped stores.
            if (!enqueue(layer, n_tok, groups_, (int64_t) dst_.size(), err, true) ||
                !check(cudaStreamSynchronize(stream_), "direct output verify sync", err, device_)) return false;
            if (std::memcmp(output_reference_.data(), h_out_, bytes) != 0) {
                err = "remote kernel writeback differs bitwise from direct output at layer " + std::to_string(layer);
                return false;
            }
            ++output_checks_;
        }
    }
    if (async_active_ && ((RemoteTransferMeta*) (meta + 1))->verify) {
        auto* transfer = (RemoteTransferMeta*) (meta + 1);
        void* mapped_ready = nullptr;
        if (!check(cudaHostGetDevicePointer(&mapped_ready, ready, 0), "map diagnostic completion", err, device_)) return false;
        transfer->ready = (uint64_t) mapped_ready;
        if (!check(cudaMemcpyAsync(d_transfer_, transfer, sizeof(*transfer), cudaMemcpyHostToDevice, stream_),
                   "publish checked metadata", err, device_) || !copy_output(n, err)) return false;
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    // WDDM can defer an asynchronous batch until the next query/sync. Submit
    // before the CPU pool runs, so the helper can compute beside it.
    static const bool eager = [] {
        const char* v = std::getenv("STRATA_REMOTE_EAGER_SUBMIT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (eager || async_active_) {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status != cudaSuccess && status != cudaErrorNotReady &&
            !check(status, "submit", err, device_)) return false;
    }
    return true;
}

bool RemoteExperts::enqueue(int64_t layer, int64_t n_tok, int64_t cap_groups,
                            int64_t cap_entries, std::string& err, bool direct_output) {
    const bool reduce = remote_opt_ && remote_opt_->active();
    const bool staged_output = output_kernel_ && !direct_output && !reduce;
    const cudaStream_t s = stream_;
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : direct_io_ ? sizeof(RemoteTransferMeta) : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& fmt = lay.fmt[(size_t) layer];
        const bool shared_input = direct_active_ && direct_input_ && !direct_output;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        if (shared_input && !copy_input(n_tok, err)) return false;
#endif
        const float* xe = shared_input ? d_x_ : zero_copy_ ? z_x_ : d_x_;
        if (fmt.had_gu) {   // APR: Hadamard-folded gate/up take H (s * x)
            strata::kernels::hadamard_rows(xe, d_xr_, n_tok, H, fmt.had_block, d_sx_, s);
            xe = d_xr_;
        }
        strata::kernels::quantize_q8_1_rows(xe, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        if (fmt.had_d) { L.had_block = fmt.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               cap_groups, cap_entries, d_q8_, d_scratch_, zero_copy_ && !reduce && !staged_output ? z_out_ : d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                        cap_groups, cap_entries, d_q8_, d_scales_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) d_meta_ + 1, err)) return false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    } else if (staged_output) {
        if (!copy_output(cap_entries, err)) return false;
#endif
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    return true;
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (group_id_.empty() || async_active_) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    // Spin on the stream like PeerExperts::finish instead of the blocking sync: the layer waits for this helper
    // on the CPU pool's critical path, and a blocking sync's wake-up costs more than the helper's small batch
    // takes - on HIP especially, where the CUDA branch's cudaInitDevice(ScheduleSpin) does not exist and the
    // device keeps its default policy.  STRATA_REMOTE_SPIN=0 keeps the blocking sync (the same switch the CUDA
    // branch reads).
#if defined(STRATA_USE_HIP)   // HIP only: the CUDA branch's cudaInitDevice(ScheduleSpin) already makes the sync spin
    static const bool spin = [] {
        const char* v = std::getenv("STRATA_REMOTE_SPIN");
        return !(v != nullptr && v[0] == '0');
    }();
    if (spin) {
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!check(e, "finish", err, device_)) return false;
    } else if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) {
        return false;
    }
#else
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
#endif
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    if (direct_active_) return true;
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

bool RemoteExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (device_ < 0 || read_stream_ == nullptr || n == 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    std::lock_guard<std::mutex> lock(read_mu_);
    bool queued = false;
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        const int32_t slot = cache_.slot_of(layers[i], experts[i]);
        if (slot < 0) continue;
        const size_t bytes = (size_t) lay.blob_bytes(layers[i]);
        if (!check(cudaMemcpyAsync(dst[i], cache_.device_slot(slot), bytes, cudaMemcpyDeviceToHost, read_stream_),
                   "cache read", err, device_)) {
            // Nothing may still be writing a caller's buffer when this returns.  If the drain succeeds, the
            // hits queued before the failure are complete and the caller may keep them; if it does not,
            // forget every hit and let the caller refill all of them from its fallback.
            if (cudaStreamSynchronize(read_stream_) != cudaSuccess) std::fill(hit, hit + n, 0);
            return false;
        }
        hit[i] = 1;
        queued = true;
    }
    if (queued && !check(cudaStreamSynchronize(read_stream_), "cache read wait", err, device_)) {
        std::fill(hit, hit + n, 0);   // a failed wait means no copied buffer may be trusted
        return false;
    }
    return true;
}

bool RemotePrefillSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst != nullptr && from_helper(layer, expert)) {
        const int32_t l = (int32_t) layer, e = (int32_t) expert;
        uint8_t* d = dst;
        std::string err;
        for (RemoteExperts* h : helpers_) {
            uint8_t got = 0;
            const bool ok = h->copy_cached_blobs(&l, &e, &d, 1, &got, err);
            if (got) {
                helper_blobs_.fetch_add(1, std::memory_order_relaxed);
                helper_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer),
                                        std::memory_order_relaxed);
                return true;
            }
            if (!ok) break;
        }
        warn(err);
    }
    return fallback_.copy_blob(layer, expert, dst);
}

bool RemotePrefillSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    if (helpers_.empty() || n == 0) return fallback_.copy_blobs(layers, experts, dst, n);
    thread_local std::vector<int32_t> hl, he;
    thread_local std::vector<uint8_t*> hd;
    thread_local std::vector<size_t> idx;
    thread_local std::vector<uint8_t> got;
    thread_local std::vector<uint8_t> served;
    thread_local std::vector<int32_t> ml, me;
    thread_local std::vector<uint8_t*> md;
    thread_local std::vector<int> which;
    which.assign(n, -1);
    served.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        for (size_t k = 0; k < helpers_.size(); ++k)
            if (helpers_[k]->holds(layers[i], experts[i])) { which[i] = (int) k; break; }
    }
    // One stream batch per helper, so several hits pay one wait; the misses go to the file source in one
    // call, which keeps its own batched-read merging for them.
    uint64_t blobs = 0, bytes = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t k = 0; k < helpers_.size(); ++k) {
        hl.clear(); he.clear(); hd.clear(); idx.clear();
        for (size_t i = 0; i < n; ++i)
            if (which[i] == (int) k) {
                hl.push_back(layers[i]); he.push_back(experts[i]); hd.push_back(dst[i]); idx.push_back(i);
            }
        if (hl.empty()) continue;
        got.assign(hl.size(), 0);
        std::string err;
        if (!helpers_[k]->copy_cached_blobs(hl.data(), he.data(), hd.data(), hl.size(), got.data(), err)) warn(err);
        for (size_t j = 0; j < got.size(); ++j)
            if (got[j]) {
                served[idx[j]] = 1;
                ++blobs;
                bytes += lay.blob_bytes(hl[j]);
            }
    }
    if (blobs != 0) {
        helper_blobs_.fetch_add(blobs, std::memory_order_relaxed);
        helper_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    ml.clear(); me.clear(); md.clear();
    for (size_t i = 0; i < n; ++i)
        if (!served[i]) { ml.push_back(layers[i]); me.push_back(experts[i]); md.push_back(dst[i]); }
    if (ml.empty()) return true;
    return fallback_.copy_blobs(ml.data(), me.data(), md.data(), ml.size());
}

void RemotePrefillSource::warn(const std::string& err) {
    if (err.empty() || warned_.exchange(true)) return;
    std::fprintf(stderr, "prefill: a helper cache read failed (%s); the file source fills that blob instead\n",
                 err.c_str());
}
} // namespace strata::core
