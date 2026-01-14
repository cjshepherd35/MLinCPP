#include <time.h>
#include <algorithm>
#include <iostream>
#include  <random>
#include <chrono>
#include <map>
#include "../../nnfromscratchfrompyversion.cpp"


int main()
{
    srand(69);
    float lr = 0.1;
    int n_inputs = 2;
    int  n_neurons = 2;
    int num_classes = 1;
    int n_samples = 4;
    int num_iters = 201;
    int check_iter = 50;
    
    Lora ld1(n_inputs, n_neurons, true, n_samples, 2);
    LayerDense ld2(n_neurons, num_classes, true, n_samples);
    // LayerDense ld3(n_neurons, num_classes, false, n_samples);
    Optimizer_SGD opt(lr);
    Relu_Activation relu(n_samples, n_neurons);
    // Relu_Activation relu2(n_samples, n_neurons);
    Sigmoid sig(n_samples,num_classes);
    BinaryCrossentropy_Loss loss(n_samples, num_classes);
    
    //Activation_softmax soft;
   double td[] = {0,0,0,
                  0,1,1,
                  1,0,1,
                  1,1,1
                  };
    
   

    Mat tri = mat_alloc(4,2);
    Mat tro = mat_alloc(4,1);
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            if(j  < 2)
            {
                MAT_AT(tri, i, j) = td[i*3+j];
            }
            else
            {
                MAT_AT(tro, i, j-2) = td[i*3+j];
            }
        }
    }


    for (size_t j = 0; j < num_iters; j++)
    {
       
        ld1.forward(tri);
        ld2.forward(ld1.output);
        sig.forward(ld2.output);

        loss.forward(sig.output, tro);

        if (j%check_iter ==  0)
        {
            printf("soft out\n");
            mat_print(sig.output);
            
            printf("true val:\n");
            mat_print(tro);
        }

        loss.backward(sig.output, tro);
        sig.backward(loss.dinputs);
        ld2.backward(sig.dinputs);
        ld1.backward(ld2.dinputs);
        
        opt.update_params(ld2);
        opt.update_params(ld1);
       
    }

    return 0;
}