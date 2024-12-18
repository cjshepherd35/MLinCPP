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

Mat mat_row(Mat m, size_t row)
{
    return (Mat){ 
        .rows = 1, 
        .cols = m.cols, 
        .stride = m.stride, 
        .es = &MAT_AT(m, row, 0)
        };
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