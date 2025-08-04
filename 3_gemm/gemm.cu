#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>


void randomMatrix (int m, int n,float * mat){
    for (int row=0; row<m; row++){
        for (int col=0; col<n;col++){
            
        }
    }
}


void main(){
    const int m = 2048;
    const int n = 2048;
    const int k = 2048;
    const size_t matrix_a = m * k * sizeof(float);
    const size_t matrix_b = k * n * sizeof(float);
    const size_t matrix_c = m * n * sizeof(float); 
}