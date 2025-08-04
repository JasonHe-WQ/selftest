#include <cuda.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>

const unsigned int WarpSize =32;

__device__ float warp_reducer(float sum){
    sum += __shfl_down_sync(0xffffffff,sum,16);
    sum += __shfl_down_sync(0xffffffff,sum,8);
    sum += __shfl_down_sync(0xffffffff,sum,4);
    sum += __shfl_down_sync(0xffffffff,sum,2);
    sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}
template <unsigned int blocksize>
__global__ void warp_reduce_launcher( const float *in_data ,float *out_data, const int size ){
    const unsigned int gtid = blockIdx.x * blocksize + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int totalThreadNum = gridDim.x * blocksize;
    const unsigned int warpId = tid / WarpSize;
    const unsigned int laneId = tid & (WarpSize-1);
    const unsigned int warpNumber = blocksize/WarpSize;
    __shared__ float warpSum[warpNumber];
    float sum = 0;
    for (unsigned int i = gtid; i<size; i+=totalThreadNum){
        sum += in_data[i];
    }
    sum = warp_reducer(sum);

    if ( laneId==0 ){
        warpSum[warpId] = sum;
    }
    __syncthreads();
    sum = (laneId < (blocksize/WarpSize))? warpSum[laneId]:0.0f;
    if (warpId == 0) {
        sum = warp_reducer(sum);
        if (laneId == 0) {
            // printf("Block ID: %d, Final Sum: %f\n", blockIdx.x, sum);
            out_data[blockIdx.x] = sum;
        }
    }

}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}


int main(){
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    constexpr int blockSize = 1024;
    const int totalThreadsToLaunch = N / 8;
    int GridSize = (totalThreadsToLaunch + blockSize - 1) / blockSize; // 安全的计算方式
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 2.0f;
    }

    float groudtruth = N * 2.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    warp_reduce_launcher<blockSize><<<Grid,Block>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %f \n", GridSize, groudtruth);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_warp_level latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}


