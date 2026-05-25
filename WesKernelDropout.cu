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