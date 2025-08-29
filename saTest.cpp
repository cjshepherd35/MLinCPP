#include <iostream>
#include "nnfromscratchfrompyversion.cpp"
#include <fstream>
#include <time.h>
#include <random>



int main()
{
    int batch_size = 32;
    int seq_l = 64;
    int embed_dim = 128;
    int num_heads = 1;
    Tensor m = tensor_alloc(batch_size, seq_l, embed_dim);
    tensor_rand(m, -1, 1);
    AttentionHead sa(embed_dim, batch_size, seq_l, embed_dim);

    sa.forward(m);
    sa.backward(sa.output);
    // for (size_t i = 0; i < 100; i++)
    // {
    //     sa.forward(m);
    //     sa.backward(sa.output);
    // }
    
    
    return 0;
}