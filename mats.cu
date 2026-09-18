#include <stdio.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cuda_runtime.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA! from here\n");
}

#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h> 
#include <random>
#include <cmath>
#include <chrono>

//structure for making matrices
struct Mat 
{
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
};

#define MAT_AT(m, i, j) m.es[i*(m).stride+j]



float rand_float();
float sigmoidf(float x);


Mat mat_alloc(size_t rows, size_t cols);
void  mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_col(Mat m, size_t col);
void mat_fill(Mat m, float x);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias);
void GPU_mat_dot(Mat dst, Mat a, Mat b);
void GPU_mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias);
void GPU_mat_dot_relu(Mat dst, Mat mmout, Mat a, Mat b);
void GPU_mat_dot_bias_relu(Mat dst, Mat mmout, Mat a, Mat b, Mat bias);
void mat_sum(Mat dst, Mat a);
void mat_sig( Mat m);
void mat_print(Mat m);
void mat_free(Mat m);
void GPU_dbias_calc(Mat dbias, Mat dvals);
void gpu_softmax_func(Mat output, Mat input);

float rand_float()
{
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x)
{

    return 1.f / (1.f + expf(-x));
}

// Mat stuff
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (float*) malloc(sizeof(*m.es)*rows*cols);
    assert(m.es != NULL);
    return m;

}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_zeroes(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = 0;
        }
    }
}


void  mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float()*(high-low) + low;
        }
    }
}

__global__ void device_mat_dot(float *dst, float *A, float *B, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int l = 0; l < K; l++) {
            sum += A[row * K + l] * B[l * N + col];
        }
        dst[row*N+col] = sum;
    }
    // printf("finished hello");
}

void cuda_mat_dot(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    int Block_size = 32;
    int size_a = a.rows*a.cols*sizeof(float);
    int size_b = b.rows*b.cols*sizeof(float);
    int size_dst = dst.rows*dst.cols*sizeof(float);

    float *d_a,*d_b, *d_dst;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b,size_b);
    cudaMalloc(&d_dst,size_dst);

    cudaMemcpy(d_a, a.es, size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.es, size_b, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dst, dst.es, size_dst, cudaMemcpyHostToDevice);
    
    dim3 blockDim(Block_size, Block_size);
    dim3 gridDim((dst.cols + Block_size - 1)/ Block_size, (dst.rows+Block_size-1)/Block_size);
    //dim3 gridDim(1,1);
    device_mat_dot<<<gridDim, blockDim>>>(d_dst,d_a,d_b, dst.rows, a.cols, dst.cols);
    cudaDeviceSynchronize();
    
    //copy back to cpu memory
    cudaMemcpy(dst.es, d_dst, size_dst, cudaMemcpyDeviceToHost);
    
    //free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_dst);
}


void mat_dot(Mat dst, Mat a, Mat b)
{
    // std::cout << "dst.rows " << dst.rows << " dst.cols " << dst.cols <<std::endl;
    // std::cout << "a.rows " << a.rows << " a.cols " << a.cols <<std::endl;
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    size_t n = a.cols;
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,  i,j) = 0;
           for (size_t k = 0; k < n; k++)
           {
                MAT_AT(dst,i,j) += MAT_AT(a, i,k)*MAT_AT(b,k,j);
           } 
        }  
    }
}


void mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    size_t n = a.cols;
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,  i,j) = 0;
           for (size_t k = 0; k < n; k++)
           {
                MAT_AT(dst,i,j) += MAT_AT(a, i,k)*MAT_AT(b,k,j);
           } 
           MAT_AT(dst, i, j) += MAT_AT(bias, 0, j);
        }  
    }

}

__global__ void gpu_matdot(float* a, float* b, float* c, int m, int k, int n)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < m && col < n)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < k; i++)
        {
            sum += a[row*k+i] * b[i*n+col];
        }
        c[row*n+col] = sum;
    }
}


void GPU_mat_dot(Mat dst, Mat a, Mat b)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    assert(a.cols == b.rows);
    int block_size = 32;
    int m = a.rows, n = b.cols, k = a.cols;
    int asize = m*k*sizeof(float);
    int bsize = k*n*sizeof(float);
    int csize = m*n*sizeof(float);

   
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, asize);
    cudaMalloc(&d_b, bsize);
    cudaMalloc(&d_c, csize);

    //copy memory to device
    cudaMemcpy(d_a, a.es, asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.es, bsize, cudaMemcpyHostToDevice);


    dim3 blockDim(block_size, block_size);
    dim3 gridDim((n+block_size-1)/block_size, (m+block_size-1)/block_size);


    //launch kernel
    gpu_matdot<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, k, n);

    // cudaMemcpy(h, d_a, asize,  cudaMemcpyDeviceToHost);
    cudaMemcpy(dst.es, d_c, csize,  cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}



__global__ void gpu_matdot_bias(float *a, float *b, float *c, float* bias, int m, int k, int n)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < m && col < n)
    {
        float sum = bias[col];
        for (size_t i = 0; i < k; i++)
        {
            sum += a[row*k+i] * b[i*n+col];
        }
        c[row*n+col] = sum;
    }
}

#define BLOCK_SIZE 32


__global__ void device_fast_mat_dot(float *dst, float *A, float *B, int M, int K, int N)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (row < M && m * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + m * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && m * BLOCK_SIZE + ty < K)
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        dst[row * N + col] = sum;
    }
}

void cuda_fast_mat_dot(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    int size_a = a.rows*a.cols*sizeof(float);
    int size_b = b.rows*b.cols*sizeof(float);
    int size_dst = dst.rows*dst.cols*sizeof(float);

    float *d_a,*d_b, *d_dst;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b,size_b);
    cudaMalloc(&d_dst,size_dst);

    cudaMemcpy(d_a, a.es, size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.es, size_b, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dst.cols + BLOCK_SIZE - 1)/ BLOCK_SIZE, (dst.rows+BLOCK_SIZE-1)/BLOCK_SIZE);
    device_fast_mat_dot<<<gridDim, blockDim>>>(d_dst,d_a,d_b, dst.rows, a.cols, dst.cols);
    cudaDeviceSynchronize();
    
    //copy back to cpu memory
    cudaMemcpy(dst.es, d_dst, size_dst, cudaMemcpyDeviceToHost);
    
    //free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_dst);
}



void GPU_mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    assert(a.cols == b.rows);
    assert(bias.cols == dst.cols);
    int block_size = 32;
    int m = a.rows, n = b.cols, k = a.cols;
    int asize = m*k*sizeof(float);
    int bsize = k*n*sizeof(float);
    int csize = m*n*sizeof(float);
    int biassize = n*sizeof(float);
   
    float *d_a, *d_b, *d_c, *d_bias;
    
    cudaMalloc(&d_a, asize);
    cudaMalloc(&d_b, bsize);
    cudaMalloc(&d_c, csize);
    cudaMalloc(&d_bias, biassize);

    //copy memory to device
    cudaMemcpy(d_a, a.es, asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.es, bsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.es, biassize, cudaMemcpyHostToDevice);


    dim3 blockDim(block_size, block_size);
    dim3 gridDim((n+block_size-1)/block_size, (m+block_size-1)/block_size);


    //launch kernel
    gpu_matdot_bias<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_bias, m, k, n);

    // cudaMemcpy(h, d_a, asize,  cudaMemcpyDeviceToHost);
    cudaMemcpy(dst.es, d_c, csize,  cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_bias);
}


__global__ void fast_mat_dot_relu(float *dst, float *A, float *B, int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (row < M && m * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + m * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && m * BLOCK_SIZE + ty < K)
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) 
    {
        dst[row * N + col] = std::max(0.f, sum);
    }
}

__global__ void fast_mat_dot_bias_relu(float *dst, float *A, float *B, float* bias, int M, int N, int K)
{
     int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (row < M && m * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + m * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && m * BLOCK_SIZE + ty < K)
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        dst[row * N + col] = std::max(0.f,bias[col] + sum);
    }
}


void gpu_mat_dot_relu(Mat dst, Mat a, Mat b)
{
    int arows = a.rows;
    int acols = a.cols;
    int brows = b.rows;
    int bcols = b.cols;
    

    float *d_a, *d_b, *d_dst;

    cudaMalloc(&d_a, arows*acols*sizeof(float));
    cudaMalloc(&d_b, brows*bcols*sizeof(float));
    cudaMalloc(&d_dst, arows*bcols*sizeof(float));

    cudaMemcpy(d_a, a.es, arows*acols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.es, brows*bcols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dst.cols + BLOCK_SIZE -1)/BLOCK_SIZE, (dst.rows + BLOCK_SIZE-1)/BLOCK_SIZE);
    fast_mat_dot_relu<<<gridDim, blockDim>>>(d_dst, d_a, d_b, arows, bcols, acols);

    cudaMemcpy(dst.es, d_dst, dst.rows*dst.cols*sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_dst);

}


void gpu_mat_dot_bias_relu(Mat dst, Mat a, Mat b, Mat bias)
{
    int arows = a.rows;
    int acols = a.cols;
    int brows = b.rows;
    int bcols = b.cols;
    

    float *d_a, *d_b, *d_bias, *d_dst;

    cudaMalloc(&d_a, arows*acols*sizeof(float));
    cudaMalloc(&d_b, brows*bcols*sizeof(float));
    cudaMalloc(&d_bias, bias.rows*bias.cols*sizeof(float));
    cudaMalloc(&d_dst, arows*bcols*sizeof(float));

    cudaMemcpy(d_a, a.es, arows*acols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.es, brows*bcols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.es, bias.rows*bias.cols*sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dst.cols + BLOCK_SIZE -1)/BLOCK_SIZE, (dst.rows + BLOCK_SIZE-1)/BLOCK_SIZE);
    fast_mat_dot_bias_relu<<<gridDim, blockDim>>>(d_dst, d_a, d_b, d_bias, arows, bcols, acols);

    cudaMemcpy(dst.es, d_dst, dst.rows*dst.cols*sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_bias);
    cudaFree(d_dst);
}



//should also probably work as dbias = dval because both should be one row of neurons.
__global__ void dbias_calc_kernel(float *dbias, float *dvals, int rows, int dvalcols)
{
    int row = blockDim.x *blockIdx.x + threadIdx.x;

    if (row < dvalcols)
    {
        dbias[row] = 0.f;
        for (size_t i = 0; i < dvalcols; i++)
        {
            dbias[row] += dvals[dvalcols*row + i];
        }
        
    }
    
}

void GPU_dbias_calc(Mat dbias, Mat dvals)
{
    float *d_dbias, *d_dvals;
    cudaMalloc(&d_dbias, dbias.rows*dbias.cols*sizeof(float));
    cudaMalloc(&d_dvals, dvals.rows*dvals.cols*sizeof(float));

    cudaMemcpy(d_dvals, dvals.es, dvals.rows*dvals.cols*sizeof(float), cudaMemcpyHostToDevice);
    int gridd = (dbias.rows + 31)/ 32;
    dbias_calc_kernel<<<gridd, 32>>>(d_dbias, d_dvals, dbias.rows, dvals.cols);
    
    cudaMemcpy(dbias.es, d_dbias, dbias.rows*dbias.cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_dbias);
    cudaFree(d_dvals);
}


//finish this. 
void gpu_softmax_func(Mat output, Mat input)
{

}



//i just dont like seeing the red line. this function is fine
Mat mat_row(Mat m, size_t row)
{
    Mat a;
    a.rows = 1;
    a.cols = m.cols;
    a.stride = m.stride;
    a.es = m.es + row*m.stride;
    return a;
}

void  mat_copy(Mat dst, Mat src)
{
    assert(dst.cols == src.cols);
    assert(dst.rows == src.rows);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i,j) = MAT_AT(src, i,j);
        }   
    }
}

void mat_transpose(Mat transpose, Mat original)
{
    assert(original.rows == transpose.cols);
    assert(original.cols == transpose.rows);
    int rows = original.rows;
    int cols = original.cols;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MAT_AT(transpose, j, i) = MAT_AT(original, i, j);
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    assert(dst.rows ==  a.rows);
    assert(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i,j);
        }   
    }
}


void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
        
    }
    
}


void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}


void mat_free(Mat m)
{
    free(m.es);
    m.rows = 0;
    m.cols = 0;
}





void benchmark_mat_dot(int size)
{
    printf("Benchmarking with matrix size %dx%d...\n", size, size);
    
    Mat a = mat_alloc(size, size);
    Mat b = mat_alloc(size, size);
    Mat dst = mat_alloc(size, size);
    Mat cpu_res = mat_alloc(size, size);
    
    mat_rand(a, -1.0f, 1.0f);
    mat_rand(b, -1.0f, 1.0f);
    
    // CPU
    auto start = std::chrono::high_resolution_clock::now();
    mat_dot(cpu_res, a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end - start;
    printf("CPU time: %.5f s\n", cpu_time.count());
    
    // GPU Naive
    start = std::chrono::high_resolution_clock::now();
    cuda_mat_dot(dst, a, b);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_naive_time = end - start;
    printf("GPU Naive time: %.5f s\n", gpu_naive_time.count());
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            if(abs(MAT_AT(dst, i, j) - MAT_AT(cpu_res, i, j)) > 0.001)
            {
                printf("not a match\n");
                return;
            }
        }
    }

    // GPU Fast
    start = std::chrono::high_resolution_clock::now();
    cuda_fast_mat_dot(dst, a, b);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_fast_time = end - start;
    printf("GPU Fast time: %.5f s\n", gpu_fast_time.count());
   for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            if(abs(MAT_AT(dst, i, j) - MAT_AT(cpu_res, i, j)) > 0.001)
            {
                printf("not a match\n");
                return;
            }
        }
    }
   
    printf("Speedup (Naive vs CPU): %.2fx\n", cpu_time.count() / gpu_naive_time.count());
    printf("Speedup (Fast vs CPU): %.2fx\n", cpu_time.count() / gpu_fast_time.count());
    printf("Speedup (Fast vs Naive): %.2fx\n", gpu_naive_time.count() / gpu_fast_time.count());
    
    free(a.es);
    free(b.es);
    free(dst.es);
}