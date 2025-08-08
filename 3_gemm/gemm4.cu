#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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


template<int M_PER_BLOCK, int N_PER_BLOCK, int K_PER_BLOCK, int NUMBER_PER_THREAD>
__global__ void cuda_gemm_v4( float* A_ptr,  float* B_ptr, float* C_ptr, const int M, const int N, const int K)
{
    const unsigned int tx = threadIdx.x ;
    const unsigned int ty = threadIdx.y ;
    float *A_ptr_start = A_ptr + blockIdx.y * M_PER_BLOCK * K;
    float *B_ptr_start = B_ptr + blockIdx.x * N_PER_BLOCK;

    __shared__ float A_Block[M_PER_BLOCK][K_PER_BLOCK];
    __shared__ float B_Block[K_PER_BLOCK][N_PER_BLOCK];


    float temp[NUMBER_PER_THREAD] = {0.f};


    for (int s =0 ; s < K ; s+= K_PER_BLOCK)
    {
        FETCH_FLOAT4(A_Block[ty][tx *NUMBER_PER_THREAD ]) = FETCH_FLOAT4(A_ptr_start[ K * ty + tx * NUMBER_PER_THREAD +s ]);
        FETCH_FLOAT4(B_Block[ty][tx *NUMBER_PER_THREAD ]) = FETCH_FLOAT4(B_ptr_start[ N * (ty +s ) + tx * NUMBER_PER_THREAD  ]);

        __syncthreads();


        for (int i=0; i<NUMBER_PER_THREAD; i ++ )
        {
            for (int k =0 ; k<K_PER_BLOCK; k++) {
                temp[i] += A_Block[ty][k] * B_Block[k][tx*NUMBER_PER_THREAD + i ];
            }
        }
        __syncthreads();
    }

    float *C_Ptr_Start = C_ptr + N * M_PER_BLOCK * blockIdx.y + blockIdx.x * N_PER_BLOCK;
    for (int i =0 ; i<NUMBER_PER_THREAD; i++)
    {
        C_Ptr_Start[ty * N + tx * NUMBER_PER_THREAD + i] = temp[i];
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



    constexpr unsigned int M_NUM_PER_BLOCK = 32;
    constexpr unsigned int N_NUM_PER_BLOCK = 32;
    constexpr unsigned int K_NUM_PER_BLOCK = 32;
    constexpr unsigned int NUM_PER_THREAD = 4;
    dim3 block(8,32);
    dim3 grid(n / N_NUM_PER_BLOCK, m / M_NUM_PER_BLOCK);

    cuda_gemm_v4<M_NUM_PER_BLOCK,N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,NUM_PER_THREAD><<<grid,block>>>(mat_a_device,mat_b_device,mat_c_device,m,n,k);



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