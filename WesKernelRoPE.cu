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