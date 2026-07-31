// LayerNorm Forward.

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

// LayerNorm Backward.
#include <cuda_runtime.h>

extern "C" __global__
void LayerNormBackwardKernel(
    const float* dY,
    float* dX,
    const float* Gamma,
    const float* LNXhat,
    const float* LNInvStd,
    float* dGamma,
    float* dBeta,
    int SeqLen,
    int ModelDim)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_sum1 = shared;
    float* s_sum2 = shared + blockDim.x;

    float dy = 0.0f;
    float gamma = 0.0f;
    float xhat = 0.0f;
    float dhat = 0.0f;

    int idx = row * ModelDim + tid;

    if (row < SeqLen && tid < ModelDim) {
        dy = dY[idx];
        gamma = Gamma[tid];
        xhat = LNXhat[idx];
        dhat = dy * gamma;

        atomicAdd(&dGamma[tid], dy * xhat);
        atomicAdd(&dBeta[tid], dy);
    }

    s_sum1[tid] = (tid < ModelDim) ? dhat : 0.0f;
    s_sum2[tid] = (tid < ModelDim) ? dhat * xhat : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum1[tid] += s_sum1[tid + stride];
            s_sum2[tid] += s_sum2[tid + stride];
        }
        __syncthreads();
    }

    if (row < SeqLen && tid < ModelDim) {
        float sum1 = s_sum1[0];
        float sum2 = s_sum2[0];
        float scale = LNInvStd[row] / ModelDim;

        dX[idx] = scale * (ModelDim * dhat - sum1 - xhat * sum2);
    }
}

extern "C" __declspec(dllexport)
void LaunchLayerNormBackward(
    const float* dY,
    float* dX,
    const float* Gamma,
    const float* LNXhat,
    const float* LNInvStd,
    float* dGamma,
    float* dBeta,
    int SeqLen,
    int ModelDim)
{
    cudaMemset(dGamma, 0, ModelDim * sizeof(float));
    cudaMemset(dBeta,  0, ModelDim * sizeof(float));

    int threads = 256;
    int blocks = SeqLen;
    int sharedBytes = 2 * threads * sizeof(float);

    LayerNormBackwardKernel<<<blocks, threads, sharedBytes>>>(
        dY,
        dX,
        Gamma,
        LNXhat,
        LNInvStd,
        dGamma,
        dBeta,
        SeqLen,
        ModelDim
    );

    cudaDeviceSynchronize();
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

// AutoRegressiveMack Backward.
#include <cuda_runtime.h>

extern "C" __global__
void AutoRegressiveMaskBackwardKernel(
    float* ScoresGrad,
    int SeqLen)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SeqLen && j < SeqLen && j > i) {
        ScoresGrad[i * SeqLen + j] = 0.0f;
    }
}

extern "C" __declspec(dllexport)
void LaunchAutoRegressiveMaskBackward(
    float* ScoresGrad,
    int SeqLen)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (SeqLen + threads.x - 1) / threads.x,
        (SeqLen + threads.y - 1) / threads.y
    );

    AutoRegressiveMaskBackwardKernel<<<blocks, threads>>>(
        ScoresGrad,
        SeqLen
    );

    cudaDeviceSynchronize();
}

// RoPE Forward.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void RoPEForwardKernel(
    float* H,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    if (idx >= TotalPairs)
        return;

    // Decompose the linear index into:
    //   row  = sequence position
    //   head = attention head
    //   pair = adjacent RoPE pair within that head
    int pair = idx % PairsPerHead;

    int temp = idx / PairsPerHead;
    int head = temp % NumHeads;
    int row  = temp / NumHeads;

    float angle = static_cast<float>(row) * InvFreq[pair];
    float c = cosf(angle);
    float s = sinf(angle);

    // H is physically laid out as:
    //
    // H[row, head * HeadDim + dimension]
    //
    // RowStride will normally equal ModelDim.
    int base =
        row  * RowStride +
        head * HeadDim +
        pair * 2;

    float x0 = H[base];
    float x1 = H[base + 1];

    H[base]     = x0 * c - x1 * s;
    H[base + 1] = x0 * s + x1 * c;
}

extern "C" __declspec(dllexport)
void LaunchRoPEForward(
    float* H,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    if (H == nullptr ||
        InvFreq == nullptr ||
        SeqLen <= 0 ||
        NumHeads <= 0 ||
        HeadDim <= 0 ||
        RowStride <= 0 ||
        (HeadDim & 1) != 0)
    {
        return;
    }

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    int Threads = 256;
    int Blocks = (TotalPairs + Threads - 1) / Threads;

    RoPEForwardKernel<<<Blocks, Threads>>>(
        H,
        InvFreq,
        SeqLen,
        NumHeads,
        HeadDim,
        RowStride
    );
}

// RoPE Backward.

extern "C" __global__
void RoPEBackwardKernel(
    float* dH,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    if (idx >= TotalPairs)
        return;

    int pair = idx % PairsPerHead;

    int temp = idx / PairsPerHead;
    int head = temp % NumHeads;
    int row  = temp / NumHeads;

    float angle = static_cast<float>(row) * InvFreq[pair];

    float c = cosf(angle);
    float s = sinf(angle);

    int base =
        row  * RowStride +
        head * HeadDim +
        pair * 2;

    float g0 = dH[base];
    float g1 = dH[base + 1];

    // Backpropagation applies the transpose/inverse rotation:
    //
    // [ dx0 ]   [  cos(a)  sin(a) ] [ dg0 ]
    // [ dx1 ] = [ -sin(a)  cos(a) ] [ dg1 ]
    dH[base]     =  g0 * c + g1 * s;
    dH[base + 1] = -g0 * s + g1 * c;
}

extern "C" __declspec(dllexport)
void LaunchRoPEBackward(
    float* dH,
    const float* InvFreq,
    int SeqLen,
    int NumHeads,
    int HeadDim,
    int RowStride)
{
    if (dH == nullptr ||
        InvFreq == nullptr ||
        SeqLen <= 0 ||
        NumHeads <= 0 ||
        HeadDim <= 0 ||
        RowStride <= 0 ||
        (HeadDim & 1) != 0)
    {
        return;
    }

    int PairsPerHead = HeadDim / 2;
    int TotalPairs = SeqLen * NumHeads * PairsPerHead;

    int Threads = 256;
    int Blocks = (TotalPairs + Threads - 1) / Threads;

    RoPEBackwardKernel<<<Blocks, Threads>>>(
        dH,
        InvFreq,
        SeqLen,
        NumHeads,
        HeadDim,
        RowStride
    );
}

// DropOut.

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

// Dropout Backward.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ unsigned int HashUInt(unsigned int x)
{
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

__device__ float Random01(uint64_t seed, int idx)
{
    unsigned int x = (unsigned int)(seed ^ (uint64_t)idx);
    x = HashUInt(x);
    return (x & 0x00FFFFFF) / 16777216.0f;
}

extern "C" __global__
void DropoutBackwardKernel(
    float* dX,
    int N,
    float DropProb,
    uint64_t Seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float r = Random01(Seed, idx);

        if (r < DropProb)
            dX[idx] = 0.0f;
        else
            dX[idx] = dX[idx] / (1.0f - DropProb);
    }
}

extern "C" __declspec(dllexport)
void LaunchDropoutBackward(
    float* dX,
    int N,
    float DropProb,
    uint64_t Seed)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    DropoutBackwardKernel<<<blocks, threads>>>(
        dX,
        N,
        DropProb,
        Seed
    );

    cudaDeviceSynchronize();
}

// ReLU Forward.

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

// ReLU Backward.

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

// Softmax Forward Strided, June 13 2026.

#include <math.h>
#include <float.h>

extern "C" __declspec(dllexport)
__global__ void SoftmaxForwardStridedKernel(
    const float* In,
    float* Out,
    int Rows,
    int Cols,       // logical vocab size: nVocab
    int RowStride,  // physical row stride: DimVocab
    float Temperature)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    if (row >= Rows)
        return;

    if (Temperature <= 0.0f)
        Temperature = 1.0f;

    int base = row * RowStride;

    // 1. Find max logit for numerical stability.
    float localMax = -FLT_MAX;

    for (int col = tid; col < Cols; col += blockDim.x) {
        float x = In[base + col];
        if (x > localMax)
            localMax = x;
    }

    shared[tid] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared[tid + stride] > shared[tid])
                shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    float maxVal = shared[0];

    // 2. Compute exp sum.
    float localSum = 0.0f;

    for (int col = tid; col < Cols; col += blockDim.x) {
        float e = expf((In[base + col] - maxVal) / Temperature);
        localSum += e;
    }

    shared[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    float sumVal = shared[0];

    // 3. Write normalized probabilities.
    for (int col = tid; col < Cols; col += blockDim.x) {
        float e = expf((In[base + col] - maxVal) / Temperature);
        Out[base + col] = e / sumVal;
    }
}

extern "C" __declspec(dllexport)
void LaunchSoftmaxForwardStrided(
    const float* dIn,
    float* dOut,
    int Rows,
    int Cols,
    int RowStride,
    float Temperature)
{
    int threads = 256;
    int blocks = Rows;
    int sharedBytes = threads * sizeof(float);

    SoftmaxForwardStridedKernel<<<blocks, threads, sharedBytes>>>(
        dIn,
        dOut,
        Rows,
        Cols,
        RowStride,
        Temperature);
}

// Softmax Backward, strided.

#include <cuda_runtime.h>

extern "C" __global__
void SoftmaxBackwardStridedKernel(
    const float* Y,
    const float* dY,
    float* dX,
    int Rows,
    int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // 1. local dot = sum(dY * Y)
    float localDot = 0.0f;

    for (int col = tid; col < D; col += blockDim.x) {
        int idx = row * D + col;
        localDot += dY[idx] * Y[idx];
    }

    shared[tid] = localDot;
    __syncthreads();

    // reduce dot
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    float dot = shared[0];

    // 2. dx = y * (dy - dot)
    for (int col = tid; col < D; col += blockDim.x) {
        int idx = row * D + col;
        dX[idx] = Y[idx] * (dY[idx] - dot);
    }
}

extern "C" __declspec(dllexport)
void LaunchSoftmaxBackward(
    const float* Y,
    const float* dY,
    float* dX,
    int Rows,
    int D)
{
    int threads = 256;
    int sharedBytes = threads * sizeof(float);

    SoftmaxBackwardStridedKernel<<<Rows, threads, sharedBytes>>>(
        Y, dY, dX, Rows, D
    );

    cudaDeviceSynchronize();
}

// CE Gradient Strided from Probabilities.
extern "C" __declspec(dllexport)
__global__ void CEGradientStridedKernel(
    const float* Probs,
    float* TopGradient,
    const int* TargetTokens,
    int Rows,
    int VocabSize,
    int RowStride,
    float GradScale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = Rows * VocabSize;
    if (idx >= total)
        return;

    int row = idx / VocabSize;
    int col = idx % VocabSize;

    int target = TargetTokens[row];
    int offset = row * RowStride + col;

    // Invalid target row contributes no gradient.
    if (target < 0 || target >= VocabSize) {
        TopGradient[offset] = 0.0f;
        return;
    }

    // Cross-entropy gradient after softmax:
    //
    // dLogits = Probs
    // dLogits[target] -= 1
    //
    // GradScale normally averages the gradient over Rows.
    float g = Probs[offset];

    if (col == target)
        g -= 1.0f;

    TopGradient[offset] = g * GradScale;
}

extern "C" __declspec(dllexport)
void LaunchCEGradientStrided(
    const float* dProbs,
    float* dTopGradient,
    const int* dTargetTokens,
    int Rows,
    int VocabSize,
    int RowStride,
    float GradScale)
{
    int threads = 256;
    int total = Rows * VocabSize;
    int blocks = (total + threads - 1) / threads;

    CEGradientStridedKernel<<<blocks, threads>>>(
        dProbs,
        dTopGradient,
        dTargetTokens,
        Rows,
        VocabSize,
        RowStride,
        GradScale);
}

// CE Gradient From Probabilities.

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

// Embedding Lookup.
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

// Scatter Add.
extern "C" __global__
void AddInputEmbeddingGradKernel(
    const float* XGrad,
    float* EmbGrad,
    const int* InputTokens,
    int SeqLen,
    int ModelDim,
    int nVocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SeqLen * ModelDim;

    if (idx < total) {
        int row = idx / ModelDim;
        int col = idx - row * ModelDim;
        int tok = InputTokens[row];

        if (tok >= 0 && tok < nVocab)
            atomicAdd(&EmbGrad[tok * ModelDim + col], XGrad[idx]);
    }
}

extern "C" __declspec(dllexport)
void LaunchAddInputEmbeddingGrad(
    const float* XGrad,
    float* EmbGrad,
    const int* InputTokens,
    int SeqLen,
    int ModelDim,
    int nVocab)
{
    int threads = 256;
    int blocks = (SeqLen * ModelDim + threads - 1) / threads;

    AddInputEmbeddingGradKernel<<<blocks, threads>>>(
        XGrad, EmbGrad, InputTokens, SeqLen, ModelDim, nVocab
    );

    cudaDeviceSynchronize();
}

// Add Bias Rows.

#include <cuda_runtime.h>

extern "C" __global__
void AddBiasRowsKernel(
    float* X,
    const float* Bias,
    int Rows,
    int Cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Rows * Cols;

    if (idx < total) {
        int col = idx % Cols;
        X[idx] += Bias[col];
    }
}

extern "C" __declspec(dllexport)
void LaunchAddBiasRows(
    float* X,
    const float* Bias,
    int Rows,
    int Cols)
{
    int threads = 256;
    int blocks = (Rows * Cols + threads - 1) / threads;

    AddBiasRowsKernel<<<blocks, threads>>>(
        X,
        Bias,
        Rows,
        Cols
    );

    cudaDeviceSynchronize();
}

// Add Bias Rows Backward.
#include <cuda_runtime.h>

extern "C" __global__
void AddBiasRowsBackwardKernel(
    const float* dX,
    float* dBias,
    int Rows,
    int Cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Rows * Cols;

    if (idx < total) {
        int col = idx % Cols;
        atomicAdd(&dBias[col], dX[idx]);
    }
}

extern "C" __declspec(dllexport)
void LaunchAddBiasRowsBackward(
    const float* dX,
    float* dBias,
    int Rows,
    int Cols)
{
    int threads = 256;
    int blocks = (Rows * Cols + threads - 1) / threads;

    AddBiasRowsBackwardKernel<<<blocks, threads>>>(
        dX,
        dBias,
        Rows,
        Cols
    );

    cudaDeviceSynchronize();
}

// Clip Vector.

extern "C" __declspec(dllexport)
__global__ void ClipVectorKernel(float* X, int N, float Limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float v = X[idx];

    if (v > Limit)
        v = Limit;
    else if (v < -Limit)
        v = -Limit;

    X[idx] = v;
}

extern "C" __declspec(dllexport)
void LaunchClipVector(float* X, int N, float Limit)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ClipVectorKernel<<<blocks, threads>>>(X, N, Limit);
}

// CELossRows.
#include <cuda_runtime.h>
#include <math.h>

__global__ void CELossRows(const float* Probs, const int* Targets, float* RowLoss, int Rows, int VocabSize, int RowStride)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row >= Rows)
        return;

    int Target = Targets[Row];

    if ((Target < 0) || (Target >= VocabSize)) {
        RowLoss[Row] = NAN;
        return;
    }

    float P = Probs[Row * RowStride + Target];

    if (!isfinite(P)) {
        RowLoss[Row] = NAN;
        return;
    }

    if (P < 1.0e-12f)
        P = 1.0e-12f;
    else if (P > 1.0f)
        P = 1.0f;

    RowLoss[Row] = -logf(P);
}

extern "C" __declspec(dllexport) void LaunchCELossRows(const float* Probs, const int* Targets, float* RowLoss, int Rows, int VocabSize, int RowStride)
{
    const int Threads = 256;
    const int Blocks = (Rows + Threads - 1) / Threads;

    CELossRows<<<Blocks, Threads>>>(Probs, Targets, RowLoss, Rows, VocabSize, RowStride);
}