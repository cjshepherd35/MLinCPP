#include <stdio.h>
#include <string>

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
void mat_sum(Mat dst, Mat a);
void mat_sig( Mat m);
void mat_print(Mat m);

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
        float sum = 1.1f;
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
    int Block_size = 64;
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
    device_mat_dot<<<blockDim,gridDim>>>(d_dst,d_a,d_b, dst.rows, a.cols, dst.cols);
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

// Mat mat_row(Mat m, size_t row)
// {
//     return (Mat){ 
//         .rows = 1, 
//         .cols = m.cols, 
//         .stride = m.stride, 
//         .es = &MAT_AT(m, row, 0)
//         };
// }

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