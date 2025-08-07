#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>


void randomMatrix (int m, int n,float * mat){
    for (int row=0; row<m; row++){
        for (int col=0; col<n;col++){
            mat[row * n + col] = (float)rand() / (float)RAND_MAX;
        }
    }
}


void cpu_gemm (float* A_ptr, float* B_ptr, float* C_ptr, const int m, const int n, const int k){
    for (int row=0; row<m; row++)
    {
        for (int col=0; col<n; col++)
        {
            float tmp = 0.f;
            for (int i=0; i<k; i++)
            {
                tmp += A_ptr[row * k + i] * B_ptr[i * n + col];
            }
            C_ptr[row * n + col] = tmp;
        }
    }
}



float compareMatrix(const float* A_ptr, const float* B_ptr, const int m, const int n)
{
    float maxdiff = 0.f;
    float diff = 0.f;
    for (int row=0; row<m; row++)
    {
        for (int col=0; col<n; col++)
        {
            diff = abs(A_ptr[row * n + col] - B_ptr[row * n + col]);
            maxdiff = (diff > maxdiff) ? diff : maxdiff;
            if (maxdiff > 0.1){
                return maxdiff;
            }
        }
    }
    return maxdiff;
}


template <int BLOCKSIZE>
__global__ void cuda_gemm_v2( float* A_ptr,  float* B_ptr, float* C_ptr, const int M, const int N, const int K)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= M){
        return;
    }
    float *A_ptr_start = A_ptr + blockIdx.y * blockDim.y * K;
    float *B_ptr_start = B_ptr + blockIdx.x * blockDim.x ;
    float tmp = 0.f;

    __shared__ float A_Block [BLOCKSIZE][BLOCKSIZE];
    __shared__ float B_Block [BLOCKSIZE][BLOCKSIZE];
    for (unsigned int s =0; s< K;s += blockDim.x){
        A_Block[threadIdx.y][threadIdx.x] = A_ptr_start[threadIdx.y * K + threadIdx.x + s];
        B_Block[threadIdx.y][threadIdx.x] = B_ptr_start[threadIdx.x + threadIdx.y * N + s * N];
        __syncthreads();
        for (unsigned int i=0 ; i < BLOCKSIZE ; i++){
                tmp += A_Block[threadIdx.y][i]*B_Block[i][threadIdx.x];
            }
        __syncthreads();
    }
    C_ptr[y * N + x] = tmp;

}


int main(){
    constexpr int m = 64;
    constexpr int n = 128;
    constexpr int k = 32;
    const size_t mat_a_size = m * k * sizeof(float);
    const size_t mat_b_size = k * n * sizeof(float);
    const size_t mat_c_size = m * n * sizeof(float);
    float *mat_a_host = (float *)malloc(mat_a_size);
    float *mat_b_host = (float *)malloc(mat_b_size);
    float *mat_c_host_result = (float *)malloc(mat_c_size);
    float *mat_c_device_result = (float *)malloc(mat_c_size);
    randomMatrix(m, k, mat_a_host);
    randomMatrix(k, n, mat_b_host);
    memset(mat_c_host_result,0,mat_c_size);
    memset(mat_c_device_result,0,mat_c_size);

    float *mat_a_device, * mat_b_device, * mat_c_device;
    cudaMalloc((void**)&mat_a_device,mat_a_size);
    cudaMalloc((void**)&mat_b_device,mat_b_size);
    cudaMalloc((void**)&mat_c_device,mat_c_size);


    cudaMemcpy(mat_a_device,mat_a_host,mat_a_size,cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b_device,mat_b_host,mat_b_size,cudaMemcpyHostToDevice);

    cpu_gemm(mat_a_host, mat_b_host, mat_c_host_result, m, n, k);
    constexpr int BLOCK =16;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);

    cuda_gemm_v2<BLOCK><<<grid,block>>>(mat_a_device,mat_b_device,mat_c_device,m,n,k);



    cudaDeviceSynchronize();
    cudaMemcpy(mat_c_device_result,mat_c_device,mat_c_size,cudaMemcpyDeviceToHost);

    float diff = compareMatrix(mat_c_host_result, mat_c_device_result,m,n);
    if (diff > 0.1)
    {
        printf("diff too big: %f",diff);
    }else
    {
        printf("diff small: %f",diff);
    }

    free(mat_a_host);
    free(mat_b_host);
    free(mat_c_host_result);
    free(mat_c_device_result);
    cudaFree(mat_a_device);
    cudaFree(mat_b_device);
    cudaFree(mat_c_device);
    return 0;
}