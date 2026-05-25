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