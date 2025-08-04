#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>


template<int blocksize>
__global__ void histogram(const int *hist, int *data, int n) {

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + tid;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    __shared__ int blockHistogram[blocksize];
    blockHistogram[tid] = 0;
    __syncthreads();
    for (unsigned int i=gtid;i<n;i+= totalThreads)
    {
        int val = hist[i];
        atomicAdd(&blockHistogram[val], 1);
    }
    __syncthreads();
    atomicAdd(&data[tid], blockHistogram[tid]);
}


bool CheckResult(int *out, int* groudtruth, int N){
    for (int i = 0; i < N; i++){
        if (out[i] != groudtruth[i]) {
            printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    int *hist = (int *)malloc(N * sizeof(int));
    int *bin = (int *)malloc(256 * sizeof(int));
    int *bin_data;
    int *hist_data;
    cudaMalloc((void **)&bin_data, 256 * sizeof(int));
    cudaMalloc((void **)&hist_data, N * sizeof(int));

    for(int i = 0; i < N; i++){
        hist[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 100000;
    }

    cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 1024;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // bug1: L68的N不能传错，之前传的256，导致L19的cache[1]打印出来为0
    histogram<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    // bug2: 同bug1，L67传进去的256表示两个buffer的数据量，这个必须得精确，之前传的N，尽管只打印第1个值，但依然导致L27打印出来的值为垃圾值
    bool is_right = CheckResult(bin, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%d ", bin[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);

    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
}