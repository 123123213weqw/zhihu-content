// cuBLASLt, fp16 NxN * NxN: the heuristic's #1 choice vs. the best of timing every enumerated configuration.
// Usage: ./lt_search N [acc]
//    acc = 32 (default): fp16 inputs, fp32 accumulation, like PyTorch's default matmul and like Triton's tl.dot
//    acc = 16: fp16 accumulation (twice the tensor-core rate on these consumer cards; a different problem)
// Enumerates algo id x tile x stages x split-K x reduction scheme x swizzle (what cublasLtMatmulAlgoCap* says is
// possible), keeps the configs that pass cublasLtMatmulAlgoCheck, times each with CUDA graphs, then re-times the top
// candidates and the heuristic's #1 with longer runs.  Clocks are warmed with 5 s of the default kernel first.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("cublasLt error %d at line %d\n", (int)s_, __LINE__); exit(1);} } while (0)
#define CKC(x) do { cudaError_t s_ = (x); if (s_ != cudaSuccess) { printf("cuda error %s at line %d\n", cudaGetErrorString(s_), __LINE__); exit(1);} } while (0)

struct Ctx {
    cublasLtHandle_t lt; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t A, B, C;
    const void *a, *b, *alpha, *beta; void *c, *ws; size_t wsSize; cudaStream_t stream;
};

static cublasStatus_t mm(Ctx& x, const cublasLtMatmulAlgo_t* algo, cudaStream_t s) {
    return cublasLtMatmul(x.lt, x.op, x.alpha, x.a, x.A, x.b, x.B, x.beta, x.c, x.C, x.c, x.C, algo, x.ws, x.wsSize, s);
}

// ms per call for `inner` back-to-back calls captured in a graph, replayed `outer` times; best of `rounds`
static double graph_time(Ctx& x, const cublasLtMatmulAlgo_t* algo, int inner, int outer, int rounds) {
    for (int i = 0; i < 3; ++i) if (mm(x, algo, x.stream) != CUBLAS_STATUS_SUCCESS) return -1;
    CKC(cudaStreamSynchronize(x.stream));
    cudaGraph_t g; cudaGraphExec_t ge;
    CKC(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < inner; ++i) mm(x, algo, x.stream);
    CKC(cudaStreamEndCapture(x.stream, &g));
    cudaError_t e = cudaGraphInstantiate(&ge, g, 0);
    if (e != cudaSuccess) { cudaGetLastError(); cudaGraphDestroy(g); return -1; }
    CKC(cudaGraphLaunch(ge, x.stream)); CKC(cudaStreamSynchronize(x.stream));
    cudaEvent_t e0, e1; CKC(cudaEventCreate(&e0)); CKC(cudaEventCreate(&e1));
    double best = 1e300;
    for (int r = 0; r < rounds; ++r) {
        CKC(cudaEventRecord(e0, x.stream));
        for (int i = 0; i < outer; ++i) CKC(cudaGraphLaunch(ge, x.stream));
        CKC(cudaEventRecord(e1, x.stream)); CKC(cudaEventSynchronize(e1));
        float ms = -1.f; CKC(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, (double)ms / (inner * outer));
    }
    CKC(cudaEventDestroy(e0)); CKC(cudaEventDestroy(e1));
    CKC(cudaGraphExecDestroy(ge)); CKC(cudaGraphDestroy(g));
    return best;
}

static std::string cfg(const cublasLtMatmulAlgo_t* algo) {
    unsigned int id = 0, tile = 0, st = 0, sk = 0, swz = 0, red = 0, inner = 0, clus = 0; size_t w;
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, sizeof(st), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner), &w);
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &clus, sizeof(clus), &w);
    char buf[200];
    snprintf(buf, sizeof buf, "algo=%u tile=%u stages=%u splitK=%u red=%u swz=%u inner=%u cluster=%u", id, tile, st, sk, red, swz, inner, clus);
    return buf;
}

struct Cand { double ms; cublasLtMatmulAlgo_t algo; };

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    int accBits = argc > 2 ? atoi(argv[2]) : 32;
    int inner = N <= 512 ? 64 : N <= 1024 ? 32 : N <= 2048 ? 16 : 8;
    double flop = 2.0 * N * (double)N * N;
    auto tfs = [&](double ms) { return ms > 0 ? flop / ms / 1e9 : 0.0; };
    Ctx x; unsigned short one16 = 0x3C00, zero16 = 0x0000; float one32 = 1.f, zero32 = 0.f;
    x.alpha = accBits == 16 ? (const void*)&one16 : (const void*)&one32;
    x.beta = accBits == 16 ? (const void*)&zero16 : (const void*)&zero32;
    CK(cublasLtCreate(&x.lt));
    CKC(cudaStreamCreate(&x.stream));
    cudaDeviceProp prop; CKC(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM count %d), N=%d fp16 inputs, %d-bit accumulate\n", prop.name, prop.multiProcessorCount, N, accBits);
    size_t n = (size_t)N * N;
    CKC(cudaMalloc((void**)&x.a, n * 2)); CKC(cudaMalloc((void**)&x.b, n * 2)); CKC(cudaMalloc(&x.c, n * 2));
    CKC(cudaMemset((void*)x.a, 0x3c, n * 2)); CKC(cudaMemset((void*)x.b, 0x34, n * 2));
    x.wsSize = 64ull << 20; CKC(cudaMalloc(&x.ws, x.wsSize));
    const cublasComputeType_t ct = accBits == 16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
    const cudaDataType_t stp = accBits == 16 ? CUDA_R_16F : CUDA_R_32F;
    CK(cublasLtMatmulDescCreate(&x.op, ct, stp));
    cublasOperation_t no = CUBLAS_OP_N;
    CK(cublasLtMatmulDescSetAttribute(x.op, CUBLASLT_MATMUL_DESC_TRANSA, &no, sizeof(no)));
    CK(cublasLtMatmulDescSetAttribute(x.op, CUBLASLT_MATMUL_DESC_TRANSB, &no, sizeof(no)));
    CK(cublasLtMatrixLayoutCreate(&x.A, CUDA_R_16F, N, N, N));
    CK(cublasLtMatrixLayoutCreate(&x.B, CUDA_R_16F, N, N, N));
    CK(cublasLtMatrixLayoutCreate(&x.C, CUDA_R_16F, N, N, N));
    cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &x.wsSize, sizeof(x.wsSize)));

    // the heuristic's ranked list; #1 is the default choice
    std::vector<cublasLtMatmulHeuristicResult_t> hl(64); int returned = 0;
    CK(cublasLtMatmulAlgoGetHeuristic(x.lt, x.op, x.A, x.B, x.C, x.C, pref, hl.size(), hl.data(), &returned));
    for (auto t0 = std::chrono::steady_clock::now(); std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 5.0;)
        graph_time(x, &hl[0].algo, inner, 2, 1);
    auto precise = [&](const cublasLtMatmulAlgo_t* algo) {
        double rough = graph_time(x, algo, inner, 2, 1);
        if (rough < 0) return -1.0;
        int outer = std::max(1, std::min(400, (int)(40.0 / (inner * rough)) + 1));
        return graph_time(x, algo, inner, outer, 3);
    };
    double top = precise(&hl[0].algo);
    printf("heuristic list: %d candidates\n", returned);
    printf("  heuristic #1 (default): %8.2f TFLOPS  %s\n", tfs(top), cfg(&hl[0].algo).c_str());

    // enumerate everything
    int ids[256], nb = 0;
    CK(cublasLtMatmulAlgoGetIds(x.lt, ct, stp, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 256, ids, &nb));
    std::vector<Cand> cands; long checked = 0, valid = 0, ran = 0;
    auto tstart = std::chrono::steady_clock::now();
    const int skList[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
    for (int i = 0; i < nb; ++i) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(x.lt, ct, stp, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, ids[i], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        size_t sz = 0;
        int tiles[128], nt = 0, stages[64], ns = 0; unsigned int maxSplitK = 1, redMask = 0, maxSwz = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof(tiles), &sz) == CUBLAS_STATUS_SUCCESS) nt = (int)(sz / sizeof(int));
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages, sizeof(stages), &sz) == CUBLAS_STATUS_SUCCESS) ns = (int)(sz / sizeof(int));
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &maxSplitK, sizeof(maxSplitK), &sz);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sz);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &maxSwz, sizeof(maxSwz), &sz);
        if (nt == 0) { tiles[0] = 0; nt = 1; }
        if (ns == 0) { stages[0] = 0; ns = 1; }
        for (int ti = 0; ti < nt; ++ti) {
            unsigned int tile = (unsigned int)tiles[ti];
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
            for (int si = 0; si < ns; ++si) {
                unsigned int stg = (unsigned int)stages[si];
                cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stg, sizeof(stg));
                for (int sk : skList) {
                    if (sk > 1 && ((int)maxSplitK <= 1 || sk > (int)maxSplitK)) continue;
                    unsigned int splitk = (unsigned int)sk;
                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk));
                    unsigned int reds[2]; int nr = 0;
                    if (sk == 1) reds[nr++] = CUBLASLT_REDUCTION_SCHEME_NONE;
                    else {
                        if (redMask & CUBLASLT_REDUCTION_SCHEME_INPLACE) reds[nr++] = CUBLASLT_REDUCTION_SCHEME_INPLACE;
                        if (redMask & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) reds[nr++] = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
                        if (nr == 0) continue;
                    }
                    for (int ri = 0; ri < nr; ++ri) {
                        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reds[ri], sizeof(reds[ri]));
                        for (unsigned int swz = 0; swz <= std::min(maxSwz, 1u); ++swz) {
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz));
                            ++checked;
                            cublasLtMatmulHeuristicResult_t hr;
                            if (cublasLtMatmulAlgoCheck(x.lt, x.op, x.A, x.B, x.C, x.C, &algo, &hr) != CUBLAS_STATUS_SUCCESS) continue;
                            if (hr.workspaceSize > x.wsSize) continue;
                            ++valid;
                            double ms = graph_time(x, &algo, inner, 1, 1);
                            if (ms < 0) continue;
                            ++ran; cands.push_back({ms, algo});
                        }
                    }
                }
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tstart).count();
    std::sort(cands.begin(), cands.end(), [](const Cand& p, const Cand& q) { return p.ms < q.ms; });
    printf("enumeration: %d algo ids, %ld configs checked, %ld pass AlgoCheck, %ld timed, %.1f s\n", nb, checked, valid, ran, secs);
    // re-time the top candidates precisely
    std::vector<std::pair<double, std::string>> top15;
    for (size_t i = 0; i < cands.size() && i < 15; ++i) {
        double ms = precise(&cands[i].algo);
        if (ms > 0) top15.push_back({ms, cfg(&cands[i].algo).c_str()});
    }
    std::sort(top15.begin(), top15.end());
    // and the whole heuristic list
    std::vector<std::pair<double, int>> hlist;
    for (int i = 0; i < returned; ++i) {
        if (hl[i].state != CUBLAS_STATUS_SUCCESS) continue;
        double ms = precise(&hl[i].algo);
        if (ms > 0) hlist.push_back({ms, i});
    }
    std::sort(hlist.begin(), hlist.end());
    if (!top15.empty()) {
        printf("  best of all:      %8.2f TFLOPS  (%.3fx of default)  %s\n", tfs(top15[0].first), top / top15[0].first, top15[0].second.c_str());
        for (size_t i = 1; i < std::min<size_t>(top15.size(), 4); ++i) printf("      #%zu %8.2f TFLOPS  %s\n", i + 1, tfs(top15[i].first), top15[i].second.c_str());
        size_t worse = std::count_if(top15.begin(), top15.end(), [&](const std::pair<double, std::string>& p) { return p.first < top; });
        printf("  of the %zu re-timed candidates, %zu beat the default\n", top15.size(), worse);
        if (!cands.empty()) printf("  worst timed config: %8.2f TFLOPS\n", tfs(cands.back().ms));
    }
    if (!hlist.empty()) printf("  best inside the heuristic list: rank %d, %8.2f TFLOPS (%zu of %d ran)\n", hlist[0].second + 1, tfs(hlist[0].first), hlist.size(), returned);
    double top2 = precise(&hl[0].algo);
    printf("  re-timed heuristic #1: %8.2f TFLOPS\n", tfs(top2));
    return 0;
}
