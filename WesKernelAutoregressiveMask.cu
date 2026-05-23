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