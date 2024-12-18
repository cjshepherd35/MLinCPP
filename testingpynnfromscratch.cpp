//#include <iostream>
#include "nnfromscratchfrompyversion.cpp"
#include <time.h>

int main()
{
    srand(time(0));
    float lr = 0.01;
    int n_inputs = 2;
    int  n_neurons = 2;
    int n_output = 1;
    int n_samples = 4;
    int num_iters = 30002;

    LayerDense ld1(n_inputs, n_neurons, true, 4);
    LayerDense ld2(n_neurons, n_output, true, 4);
    Optimizer_SGD opt(lr);
    Relu_Activation relu;
    Sigmoid sig2;
    BinaryCrossentropy_Loss bl;
    //Activation_softmax soft;
    float td[] = {0,0,0,
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
    // std::cout << "input \n";
    // mat_print(tri);
    // std::cout << "y_true \n";
    // mat_print(tro);
    //Mat dvals = mat_alloc(n_samples, n_output);
    for (size_t j = 0; j < num_iters; j++)
    {
        ld1.forward(tri);
        relu.forward(ld1.output);
        ld2.forward(relu.output);
        sig2.forward(ld2.output);
        if (j % 1000 == 0)
        {
            std::cout << "output \n";
            mat_print(sig2.output);
        }
        
        bl.backward(sig2.output, tro);
        
        sig2.backward(bl.dinputs);
        ld2.backward(sig2.dinputs);
        
        relu.backward(ld2.dinputs);
        ld1.backward(relu.dinputs);

        opt.update_params(ld2);
        opt.update_params(ld1);
    }

    
    return 0;
}



