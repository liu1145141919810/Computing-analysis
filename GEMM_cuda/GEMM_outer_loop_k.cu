#include <crtdbg.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../GEMM_common/GEMM_base.h"
#include "GEMM_cuda.h"

using namespace GEMM_cuda;

__global__ void kernelOuterLoopK(const float* vMatrixA, const float* vMatrixB, float* vMatrixC,
    int vM, int vN, int vK, float vAlpha, float vBeta)
{
    //todo: add your code here
}

void GEMM_cuda::outer_loop_k_gpu(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, 
    size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
    dim3 BlockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 GridDim((vN + TILE_DIM_N - 1) / TILE_DIM_N, (vM + TILE_DIM_M - 1) / TILE_DIM_M);

    kernelOuterLoopK<<<GridDim, BlockDim >>>(vMatrixA, vMatrixB, vMatrixC, vM, vN, vK, vAlpha, vBeta);
    _ASSERTE(cudaGetLastError() == cudaSuccess);
}