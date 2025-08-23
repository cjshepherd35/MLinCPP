//seems to be a failed attempt, but it sometimes seemed like it was working.


#include "nnfromscratchfrompyversion.cpp"
#include <time.h>

int main()
{
    float lr = 0.1;
    int n_inputs = 2;
    int  n_neurons = 2;
    int n_output = 1;
    int n_samples = 4;
    int num_iters = 30000;
    int check_iter = 6000;
    float loss = 0;
    
    LayerDense ld1(n_inputs, n_neurons, true, 4);
    LayerDense ld2(n_neurons, n_output, true, 4);
    Optimizer_SGD opt(lr);
    Sigmoid sig;
    Sigmoid sig2;
    BinaryCrossentropy_Loss bl;
    //Activation_softmax soft;
    double td[] = {0,0,0,
                  0,1,1,
                  1,0,1,
                  1,1,0
                  };

    //when changing datasets, change here
    size_t stride = 3;
    size_t  n = sizeof(td)/sizeof(td[0])/stride;
    Mat tri = {
        .rows = n,
        //when changing datasets, change here
        .cols = 2,
        .stride = stride, 
        .es = td
    };

    Mat tro = {
        .rows = n,
        //when changing datasets, change here
        .cols = 1,
        .stride = stride,
        //when changing datasets, change here
        .es = td + 2
    };
    
    for (size_t j = 0; j < num_iters; j++)
    {
        ld1.forward(tri);
        sig.forward(ld1.output);
        ld2.forward(sig.output);
        sig2.forward(ld2.output);
        loss = 0;
        for (size_t i = 0; i < sig2.output.rows; i++)
        {
            loss += abs(MAT_AT(sig2.output, i,0) - MAT_AT(tro, i,0));
        }
        
        if (j % check_iter == 0)
        { 
            
            std::cout << "output \n";
            mat_print(sig2.output);
            std::cout << "y \n";
            mat_print(tro);
            std::cout << " loss " << loss << std::endl;
        }
        
       opt.update_randomly(ld2, loss);
       opt.update_randomly(ld1, loss);
        // opt.update_params(ld2);
        // opt.update_params(ld1);
    }

    
    return 0;
}
