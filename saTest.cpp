#include <iostream>
#include "nnfromscratchfrompyversion.cpp"
#include <fstream>
#include <time.h>
#include <random>



int main()
{
    int batch_size = 4;
    int seq_l = 8;
    int embed_dim = 64;
    int num_heads = 1;
    Tensor m = tensor_alloc(batch_size, seq_l, embed_dim);
    AttentionHead sa(embed_dim, batch_size, seq_l, embed_dim);
    sa.forward(m);

    return 0;
}