//this has to be done using object oriented version of matrix and tensors i think. so must be totally remade. 
#include "../nnfromscratchfrompyversion.cpp"
#include "bytepairencoding.cpp"

int n_embed = 6;
int batch_size = 2;
int seq_len = 3;
int num_iters = 1;

Tensor get_sample(Mat embedtable, std::vector<int> x_train)
{
    Tensor t = tensor_alloc(batch_size, seq_len, n_embed);
    for (size_t i = 0; i < batch_size; i++)
    {
        int randnum = rand() % (x_train.size() - seq_len);
        for (size_t j = 0; j < seq_len; j++)
        {
            for (size_t k = 0; k < n_embed; k++)
            {
                t.mats[i](j,k) = embedtable(x_train[randnum]+j,k);
            }
        }
    }
    return t;
}


int main()
{
    
    twovec d = get_data();
    int vocab_size = d.second.size();

    twovec x = get_train_data(d);
    std::vector<int> x_train = x.first;
    std::vector<int> x_test = x.second;

    Mat embedtable = mat_alloc(d.second[d.second.size()-1], n_embed);
    mat_rand(embedtable, -1.0, 1.0);

    AttentionHead head(n_embed, batch_size, seq_len, n_embed);

    for (size_t i = 0; i < num_iters; i++)
    {
        Tensor t = get_sample(embedtable, x_train);

        head.forward(t);
        head.backward(head.output);
        
        tensor_print(head.output);
        tensor_free(t);
    }
    
   
    
    return 0;
}



