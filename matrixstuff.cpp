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
    double* es;
    // This replaces the macro safely
    double& operator()(size_t r, size_t c) {
        // Optional: Add bounds checking here for debugging
        // assert(r < rows && c < cols); 
        return es[r * cols + c];
    }
};


#define MAT_AT(m, i, j) m.es[(i)*(m).stride+(j)]



double rand_double();
double sigmoid(double x);


Mat mat_alloc(size_t rows, size_t cols);
void  mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_col(Mat m, size_t col);
void mat_fill(Mat m, float x);
void mat_dot(Mat dst, Mat a, Mat b);
void rowwise_mat_dot(Mat dst, Mat a, Mat b);
void power_mat_dot(Mat dst, Mat a, Mat b, float exp);
void power_mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias, float exp);
void mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias);
void rowtocolkron(Mat dst,  Mat a, Mat b);
void rowtocolkron_bias(Mat dst, Mat a, Mat b, Mat bias);
void rowtocolkron_backwards(Mat dst, Mat dvalues, Mat layerinputs);
void mat_sum(Mat dst, Mat a);
void mat_scalertimes(Mat dst, Mat a, float scale);
void mat_elemwise_mult(Mat dst, Mat a, Mat b);
void mat_random_mask(Mat a, float percentones);
void mat_transpose(Mat transpose, Mat original);
void mat_sig( Mat m);
void mat_print(Mat m);
void mat_free(Mat& m);
void mat_half_dot(Mat dst, Mat a, Mat b);

double rand_double()
{
    return (double) rand() / (double) RAND_MAX;
}

double sigmoid(double x)
{

    return 1.0 / (1.f + exp(-x));
}

// Mat stuff
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (double*) malloc(sizeof(*m.es)*rows*cols);
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
            MAT_AT(m, i, j) = rand_double()*(high-low) + low;
        }
    }
}


void mat_dot(Mat dst, Mat a, Mat b)
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
        }  
    }
}


void mat_half_dot(Mat dst, Mat a, Mat b)
{
    assert(dst.cols % 2 == 0);
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == 2*b.cols);
    size_t n = a.cols;
    size_t dr = dst.cols/2;
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dr; j++)
        {
            MAT_AT(dst,  i,j) = 0;
            MAT_AT(dst, i, dr+j) = 0;
           for (size_t k = 0; k < n; k++)
            {
                if (k < n/2)
                {
                    MAT_AT(dst,i,j) += MAT_AT(a, i,k)*MAT_AT(b,k,j);
                }
                else
                {
                    MAT_AT(dst,i,dr+j) += MAT_AT(a, i,k)*MAT_AT(b,k,j);
                }  
            } 
        }  
    }
}


void rowwise_mat_dot(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.cols);
    assert(dst.rows ==a.rows);
    assert(dst.cols == b.rows);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i,j) += MAT_AT(a,i,k) * MAT_AT(b, j,k);
            }
        }
    }
}


void power_mat_dot(Mat dst, Mat a, Mat b, float exp)
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
                MAT_AT(dst,i,j) += MAT_AT(a, i,k)* std::pow(MAT_AT(b,k,j), exp);
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
            MAT_AT(dst,  i,j) = MAT_AT(bias, 0, j);
           for (size_t k = 0; k < n; k++)
           {
                MAT_AT(dst,i,j) += MAT_AT(a, i,k)*MAT_AT(b,k,j);
            } 
        }  
    }

}


void rowtocolkron(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.cols*b.cols);
    assert(dst.cols == a.rows);
    size_t n = a.cols;
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < b.cols; j++)
        {
           for (size_t k = 0; k < n; k++)
           {
                MAT_AT(dst,(j*a.cols+k),i) = MAT_AT(a, i,k)*MAT_AT(b,k,j);
                // std::cout << "a: " << MAT_AT(a,  i,k) << "at " << i << "," << k << " b: " << MAT_AT(b,k,j) << " at " << k << "," << j << std::endl;
                // std::cout << "= c " << MAT_AT(dst, j*a.cols+k, i) << "at " << j*a.cols+k << "," << i << std::endl;
           } 
        }  
    }
}


void rowtocolkron_bias(Mat dst, Mat a, Mat b, Mat bias)
{
     assert(a.cols == b.rows);
    assert(dst.rows == a.cols*b.cols);
    assert(dst.cols == a.rows);
    size_t n = a.cols;
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < b.cols; j++)
        {
           for (size_t k = 0; k < n; k++)
           {
                MAT_AT(dst,(j*a.cols+k),i) = MAT_AT(a, i,k)*MAT_AT(b,k,j);
                MAT_AT(dst,(j*a.cols+k),i) += MAT_AT(bias,(j*a.cols+k),i );
           } 
        }  
    }
}

void rowtocolkron_backwards(Mat dst, Mat dvalues, Mat layerinputs)
{

}


void mat_elemwise_mult(Mat dst, Mat a, Mat b)
{
    assert(a.rows == dst.rows);
    assert(a.cols == dst.cols);
    assert(b.rows == a.rows);
    assert(b.cols == a.cols);
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(a, i,j) * MAT_AT(b, i, j);
        }
    }
}


void power_mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias, float exp)
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
                MAT_AT(dst,i,j) += MAT_AT(a, i,k)* std::pow(MAT_AT(b,k,j),exp);
           } 
           MAT_AT(dst, i, j) += MAT_AT(bias, 0, j);
        }  
    }
}


void mat_scalertimes(Mat dst, Mat a, float scale)
{
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            MAT_AT(a,i,j) = MAT_AT(a,i,j) *scale;
        }
        
    }
}

void mat_random_mask(Mat a, float percentones)
{
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            float randnum = static_cast<float>(rand() / RAND_MAX);
            if(randnum > percentones)
            {
                MAT_AT(a, i, j) = 1;
            }
        }
        
    }
    
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat) { 
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
            MAT_AT(m, i, j) = sigmoid(MAT_AT(m, i, j));
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


void mat_free(Mat& m)
{
    free(m.es);
    m.es = nullptr;
    m.rows = 0;
    m.cols = 0;
    m.stride = 0;
}

