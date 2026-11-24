// The three hand-written FP32 GEMM kernels from bench.cu, exported for launching on a caller-provided stream.
#include <cuda_runtime.h>

__global__ void naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = acc;
    }
}

template<int TILE>
__global__ void tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1], Bs[TILE][TILE + 1];
    int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K) ? A[(size_t)row * K + t + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < K && col < N) ? B[(size_t)(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

#define T 64
#define RT 4
__global__ void regtile(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T + 1], Bs[T][T + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * T + ty * RT, col0 = blockIdx.x * T + tx * RT;
    float acc[RT][RT] = {};
    for (int t = 0; t < K; t += T) {
        for (int i = ty; i < T; i += 16)
            for (int j = tx; j < T; j += 16) {
                int r = blockIdx.y * T + i, c = t + j;
                As[i][j] = (r < M && c < K) ? A[(size_t)r * K + c] : 0.f;
                int r2 = t + i, c2 = blockIdx.x * T + j;
                Bs[i][j] = (r2 < K && c2 < N) ? B[(size_t)r2 * N + c2] : 0.f;
            }
        __syncthreads();
        for (int k = 0; k < T; ++k) {
            float a[RT], b[RT];
            #pragma unroll
            for (int p = 0; p < RT; ++p) { a[p] = As[ty * RT + p][k]; b[p] = Bs[k][tx * RT + p]; }
            #pragma unroll
            for (int p = 0; p < RT; ++p)
                #pragma unroll
                for (int q = 0; q < RT; ++q) acc[p][q] += a[p] * b[q];
        }
        __syncthreads();
    }
    for (int p = 0; p < RT; ++p)
        for (int q = 0; q < RT; ++q) {
            int r = row0 + p, c = col0 + q;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[p][q];
        }
}

extern "C" {
void run_naive(const float* A, const float* B, float* C, int n, void* stream) {
    dim3 blk(16, 16), grd((n + 15) / 16, (n + 15) / 16);
    naive<<<grd, blk, 0, (cudaStream_t)stream>>>(A, B, C, n, n, n);
}
void run_tiled(const float* A, const float* B, float* C, int n, void* stream) {
    dim3 blk(32, 32), grd((n + 31) / 32, (n + 31) / 32);
    tiled<32><<<grd, blk, 0, (cudaStream_t)stream>>>(A, B, C, n, n, n);
}
void run_regtile(const float* A, const float* B, float* C, int n, void* stream) {
    dim3 blk(16, 16), grd((n + 63) / 64, (n + 63) / 64);
    regtile<<<grd, blk, 0, (cudaStream_t)stream>>>(A, B, C, n, n, n);
}
}
