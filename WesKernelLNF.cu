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