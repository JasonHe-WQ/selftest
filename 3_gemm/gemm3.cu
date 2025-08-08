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


template <int BLOCKSIZE,int STRIDE>
__global__ void cuda_gemm_v3( float* A_ptr,  float* B_ptr, float* C_ptr, const int M, const int N, const int K)
{
    constexpr unsigned int STEP = BLOCKSIZE * STRIDE;
    const unsigned int tx = threadIdx.x ;
    const unsigned int ty = threadIdx.y ;
    float *A_ptr_start = A_ptr + STEP * blockIdx.y * K;
    float *B_ptr_start = B_ptr + STEP * blockIdx.x ;


    __shared__ float A_Block [STEP][STEP];
    __shared__ float B_Block [STEP][STEP];
    float tmp[STRIDE][STRIDE] = {0.f};



    for (unsigned int s =0; s< K;s += STEP){

// #pragma unroll
        for (unsigned int i =0; i < STRIDE; i++){
            for (unsigned int j =0; j< STRIDE;j++){
                A_Block[ ty + i * BLOCKSIZE][tx + j * BLOCKSIZE ] = A_ptr_start[(ty + i * BLOCKSIZE) * K + tx + j * BLOCKSIZE + s ];
                B_Block[ ty + i * BLOCKSIZE][tx + j * BLOCKSIZE ] = B_ptr_start[(ty + i * BLOCKSIZE + s) * N + tx + j * BLOCKSIZE ] ;
            }
        }
        __syncthreads();


// #pragma unroll
        for (unsigned int i =0; i < STRIDE; i++){
            for (unsigned int j =0; j< STRIDE;j++){
                for (unsigned int k=0 ; k < STEP ; k++){
                    tmp[i][j] += A_Block[ty + i * BLOCKSIZE][k] * B_Block[k][tx + j * BLOCKSIZE] ;
            }
        }}
        __syncthreads();
    }
    float *C_ptr_start = C_ptr + N * blockIdx.y * STEP + blockIdx.x * STEP;
// #pragma unroll
    for (unsigned int i =0; i < STRIDE; i++)
    {
        for (unsigned int j =0; j< STRIDE; j++)
        {
            C_ptr_start[ N *( ty + i * BLOCKSIZE) + tx + j * BLOCKSIZE ] = tmp[i][j];
        }
    }

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
    constexpr int STRIDE =2;
    constexpr int STEP = BLOCK * STRIDE;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((n + STEP - 1) / STEP, (m + STEP - 1) / STEP);

    cuda_gemm_v3<BLOCK,STRIDE><<<grid,block>>>(mat_a_device,mat_b_device,mat_c_device,m,n,k);



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