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