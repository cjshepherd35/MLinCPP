#include "matrixstuff.cpp"
#include <vector>

struct Tensor
{
    size_t depth;
    size_t rows;
    size_t cols;
    size_t stride;
    std::vector<Mat> mats;
};


Tensor  tensor_alloc(size_t depth, size_t rows, size_t cols);
void tensor_fill(Tensor t, float x);
void tensor_rand(Tensor t, float low, float high);
void tensor_print(Tensor t);
void bmm(Tensor dst, Tensor a, Mat b);
void bmm(Tensor dst, Tensor a, Tensor b);
void tensor_free(Tensor& t);


Tensor tensor_alloc(size_t depth, size_t rows, size_t cols)
{
    Tensor t;
    t.depth = depth;
    t.rows = rows;
    t.cols  = cols;
    for (size_t i = 0; i < t.depth; i++)
    {
        t.mats.push_back(mat_alloc(t.rows, t.cols));
    }
    return t;
    
}

void tensor_fill(Tensor t, float x)
{
    for (size_t i = 0; i < t.depth; i++)
    {
        mat_fill(t.mats[i], x);
    }
    
}

void tensor_rand(Tensor t, float low, float high)
{
    for (size_t i = 0; i < t.depth; i++)
    {
        mat_rand(t.mats[i], low, high);
    }
    

}

void bmm(Tensor dst, Tensor a, Mat b)
{
    assert(dst.depth == a.depth);
    
    for (size_t i = 0; i < dst.depth; i++)
    {
        mat_dot(dst.mats[i], a.mats[i], b);
    }
    
}


void tensor_print(Tensor t)
{
    for (size_t i = 0; i < t.depth; i++)
    {
        mat_print(t.mats[i]);
    }
    
}


void tensor_free(Tensor& t)
{
    for( Mat& m : t.mats)
    {
        mat_free(m);
    }
    t.mats.clear();
    t.depth = 0;
    t.rows = 0;
    t.cols = 0;
    t.stride = 0;
}