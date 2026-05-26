// LayerNormForward.

extern "C" __global__
void LayerNormForwardKernel(
    const float* InX,
    float* OutX,
    const float* Gamma,
    const float* Beta,
    float* LNXhat,
    float* LNInvStd,
    int SeqLen,
    int ModelDim,
    float EPS)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_var = shared + blockDim.x;

    float x = 0.0f;

    if (row < SeqLen && tid < ModelDim)
        x = InX[row * ModelDim + tid];

    s_sum[tid] = (tid < ModelDim) ? x : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float mean = s_sum[0] / ModelDim;

    float diff = 0.0f;
    if (tid < ModelDim)
        diff = x - mean;

    s_var[tid] = (tid < ModelDim) ? diff * diff : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_var[tid] += s_var[tid + stride];
        __syncthreads();
    }

    float invStd = rsqrtf((s_var[0] / ModelDim) + EPS);

    if (tid == 0)
        LNInvStd[row] = invStd;

    if (row < SeqLen && tid < ModelDim) {
        float xhat = diff * invStd;
        LNXhat[row * ModelDim + tid] = xhat;
        OutX[row * ModelDim + tid] =
            xhat * Gamma[tid] + Beta[tid];
    }
}

extern "C" __declspec(dllexport)
void LaunchLayerNormForward(
    const float* InX,
    float* OutX,
    const float* Gamma,
    const float* Beta,
    float* LNXhat,
    float* LNInvStd,
    int SeqLen,
    int ModelDim)
{
    int threads = 256;
    int blocks = SeqLen;
    int sharedBytes = 2 * threads * sizeof(float);

    LayerNormForwardKernel<<<blocks, threads, sharedBytes>>>(
        InX,
        OutX,
        Gamma,
        Beta,
        LNXhat,
        LNInvStd,
        SeqLen,
        ModelDim,
        1.0e-5f
    );
}

// AutoRegressiveMask.

#include <cuda_runtime.h>

extern "C" __global__
void AutoRegressiveMaskKernel(float* Scores, int SeqLen)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SeqLen && j < SeqLen && j > i) {
        Scores[i * SeqLen + j] = -1.0e30f;
    }
}

extern "C" __declspec(dllexport)
void LaunchAutoRegressiveMask(float* Scores, int SeqLen)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (SeqLen + threads.x - 1) / threads.x,
        (SeqLen + threads.y - 1) / threads.y
    );

    AutoRegressiveMaskKernel<<<blocks, threads>>>(Scores, SeqLen);
}

// RoPEForward.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void RoPEForwardKernel(
    float* H,
    const float* InvFreq,
    int SeqLen,
    int ModelDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int pair = blockIdx.x * blockDim.x + threadIdx.x;

    int NumPairs = ModelDim / 2;

    if (row < SeqLen && pair < NumPairs) {
        float angle = row * InvFreq[pair];
        float c = cosf(angle);
        float s = sinf(angle);

        int j0 = row * ModelDim + 2 * pair;
        int j1 = j0 + 1;

        float x0 = H[j0];
        float x1 = H[j1];

        H[j0] = x0 * c - x1 * s;
        H[j1] = x0 * s + x1 * c;
    }
}

extern "C" __declspec(dllexport)
void LaunchRoPEForward(
    float* H,
    const float* InvFreq,
    int SeqLen,
    int ModelDim)
{
    int NumPairs = ModelDim / 2;

    dim3 threads(16, 16);
    dim3 blocks(
        (NumPairs + threads.x - 1) / threads.x,
        (SeqLen   + threads.y - 1) / threads.y
    );

    RoPEForwardKernel<<<blocks, threads>>>(
        H,
        InvFreq,
        SeqLen,
        ModelDim
    );
}

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__
void DropoutKernel(
    float* X,
    int N,
    float DropProb,
    unsigned long long Seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        curandState state;
        curand_init(Seed, idx, 0, &state);

        float r = curand_uniform(&state);

        if (r < DropProb)
            X[idx] = 0.0f;
        else
            X[idx] = X[idx] / (1.0f - DropProb);
    }
}

// ReLUForward.
#include <cuda_runtime.h>

extern "C" __global__
void ReLUForwardKernel(
    const float* A,
    float* B,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = A[idx];
        B[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchReLUForward(
    const float* A,
    float* B,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ReLUForwardKernel<<<blocks, threads>>>(A, B, N);
}

// ReLUBackward.
#include <cuda_runtime.h>

extern "C" __global__
void ReLUBackwardKernel(
    const float* Hidden1,     // forward input
    const float* GradOut,     // Hidden2.Grad
    float* GradIn,            // Hidden1.Grad
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = Hidden1[idx];
        GradIn[idx] = (x > 0.0f) ? GradOut[idx] : 0.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchReLUBackward(
    const float* Hidden1,
    const float* GradOut,
    float* GradIn,
    int Rows,
    int Cols)
{
    int N = Rows * Cols;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ReLUBackwardKernel<<<blocks, threads>>>(
        Hidden1,
        GradOut,
        GradIn,
        N
    );

    cudaDeviceSynchronize();  // good for debugging
}

// Dropout.
extern "C" __declspec(dllexport)
void LaunchDropout(
    float* X,
    int N,
    float DropProb,
    unsigned long long Seed)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    DropoutKernel<<<blocks, threads>>>(X, N, DropProb, Seed);
}

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void SoftmaxForwardNKernel(
    const float* X,
    float* Y,
    int Rows,
    int N,
    float Temperature)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    float invT = 1.0f / Temperature;

    // 1. Find row max
    float val = -1.0e30f;

    if (row < Rows && tid < N)
        val = X[row * N + tid] * invT;

    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared[tid + stride] > shared[tid])
                shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    float maxVal = shared[0];

    // 2. Exp and sum
    float e = 0.0f;

    if (row < Rows && tid < N)
        e = expf((X[row * N + tid] * invT) - maxVal);

    shared[tid] = e;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    float invSum = 1.0f / shared[0];

    // 3. Normalize
    if (row < Rows && tid < N)
        Y[row * N + tid] = e * invSum;
}

// SoftmaxForward.
extern "C" __declspec(dllexport)
void LaunchSoftmaxForwardN(
    const float* X,
    float* Y,
    int Rows,
    int N,
    float Temperature)
{
    int threads = 256;

    // For now, require N <= 256.
    int sharedBytes = threads * sizeof(float);

    SoftmaxForwardNKernel<<<Rows, threads, sharedBytes>>>(
        X,
        Y,
        Rows,
        N,
        Temperature
    );
}

// CEGradientFromProbabilities.
#include <cuda_runtime.h>

extern "C" __global__
void CEGradientSubtractKernel(
    float* TopGradient,
    const int* TargetTokens,
    int SeqLen,
    int nVocab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SeqLen) {
        int target = TargetTokens[i];

        if (target >= 0 && target < nVocab)
            TopGradient[i * nVocab + target] -= 1.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchCEGradient(
    const float* Probs,
    float* TopGradient,
    const int* TargetTokens,
    int SeqLen,
    int nVocab)
{
    cudaMemcpy(
        TopGradient,
        Probs,
        SeqLen * nVocab * sizeof(float),
        cudaMemcpyDeviceToDevice
    );

    int threads = 256;
    int blocks = (SeqLen + threads - 1) / threads;

    CEGradientSubtractKernel<<<blocks, threads>>>(
        TopGradient,
        TargetTokens,
        SeqLen,
        nVocab
    );
}

extern "C" __global__
void EmbeddingLookupKernel(
    const float* Embeddings,
    const int* InputTokens,
    float* X,
    int SeqLen,
    int ModelDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SeqLen * ModelDim;

    if (idx < total) {
        int row = idx / ModelDim;
        int col = idx % ModelDim;
        int tok = InputTokens[row];

        X[idx] = Embeddings[tok * ModelDim + col];
    }
}

extern "C" __declspec(dllexport)
void LaunchEmbeddingLookup(
    const float* Embeddings,
    const int* InputTokens,
    float* X,
    int SeqLen,
    int ModelDim)
{
    int threads = 256;
    int blocks = (SeqLen * ModelDim + threads - 1) / threads;

    EmbeddingLookupKernel<<<blocks, threads>>>(
        Embeddings,
        InputTokens,
        X,
        SeqLen,
        ModelDim
    );
}
