
#ifndef NN_H
#define NN_H
#include <algorithm>
#include "matrixstuff.cpp"


class LayerDense
{
public:
    LayerDense(int n_inputs, int n_neurons, bool bias, int num_samples);
    void forward(Mat inputs);
    void backward(Mat dvalues);
    //~LayerDense();
    Mat output;
    Mat dinputs;
    Mat dbiases;

//private:
    Mat weights;
    Mat weights_T;
    Mat biases;
    bool layerbias;
    int n_samples;
    
    Mat dweights;
    //Mat dbiases;
    // Mat dinputs;
    Mat layerinputs;
    Mat layerinputs_T;
};
#endif

LayerDense::LayerDense(int n_inputs, int n_out, bool bias, int num_samples)
{
    n_samples = num_samples;
    layerbias = bias;
    weights = mat_alloc(n_inputs, n_out);
    mat_rand(weights, 0, 1);
    weights_T = mat_alloc(n_out, n_inputs);
    if (layerbias == true)
    {
        biases = mat_alloc(1,n_out);
        mat_rand(biases, 0,1);
        dbiases = mat_alloc(1, n_out);
    }
    dweights = mat_alloc(n_inputs, n_out);
    dinputs = mat_alloc(num_samples,n_inputs);
    layerinputs = mat_alloc(num_samples, n_inputs);
    layerinputs_T = mat_alloc(n_inputs, num_samples);
    output = mat_alloc(n_samples,n_out);
}

void LayerDense::forward(Mat inputs)
{
    layerinputs = inputs;
    if (layerbias)
    {
        mat_dot_bias(output, inputs, weights, biases);
    }
    else
    {
        mat_dot(output, inputs, weights);
    }
}

void LayerDense::backward(Mat dvalues)
{
    mat_transpose(layerinputs_T,  layerinputs);
    mat_dot(dweights, layerinputs_T, dvalues);
    
    // need to figure out what this part means......
    if (layerbias)
    {
       for (size_t i = 0; i < dbiases.rows; i++)
       {
        MAT_AT(dbiases, i, 0) = 0;
           for (size_t j = 0; j < dvalues.cols; j++)
           {
                MAT_AT(dbiases, i, 0) += MAT_AT(dvalues, i, j);
           }   
       }
    }
    //...........
    mat_transpose(weights_T, weights);
    mat_dot(dinputs, dvalues, weights_T);
}




class Relu_Activation
{
public:
    void forward(Mat inputs);
    void backward(Mat dvalues);

    Mat output;
    Mat layerinputs;
    Mat dinputs;
};

void Relu_Activation::forward(Mat inputs)
{
    layerinputs = mat_alloc(inputs.rows,inputs.cols);
    output = mat_alloc(inputs.rows, inputs.cols);
    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            MAT_AT(output,i,j) = std::max(0.f,MAT_AT(inputs,i,j));
            MAT_AT(layerinputs,i,j) = MAT_AT(inputs,i,j);
        } 
    }
}


void Relu_Activation::backward(Mat dvalues)
{
    dinputs = mat_alloc(dvalues.rows, dvalues.cols);
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            MAT_AT(dinputs,i,j) = MAT_AT(dvalues,i,j);
            if (MAT_AT(layerinputs,i,j) <=0.f)
            {
                MAT_AT(dinputs,i,j) = 0.f;
            }
        }
    }
}


class Sigmoid
{
public:
    void forward(Mat inputs);
    void backward(Mat dvalues);
    //~Sigmoid();
    Mat output;
    Mat dinputs;
};

void Sigmoid::forward(Mat inputs)
{
    output = mat_alloc(inputs.rows, inputs.cols);
    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            MAT_AT(output, i,j) = 1/(1+std::exp(-1*MAT_AT(inputs, i, j)));
        }
    }
}


void Sigmoid::backward(Mat dvalues)
{
    dinputs = mat_alloc(dvalues.rows, dvalues.cols);
    
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            MAT_AT(dinputs,i,j) = MAT_AT(dvalues,i,j) * (1 - MAT_AT(output,i,j))*MAT_AT(output,i,j);
        }
    }
}



class Activation_softmax
{
public:
    void forward(Mat inputs);
    void backward(Mat dvalues);
     Mat output;
    Mat dinputs;
};

void Activation_softmax::forward(Mat inputs)
{   
    output = mat_alloc(inputs.rows,  inputs.cols);
    Mat exp_vals = mat_alloc(inputs.rows, inputs.cols);
    Mat exp_sum = mat_alloc(inputs.rows, 1);
    Mat maxes = mat_alloc(inputs.rows, 1);
    for (size_t i = 0; i < inputs.rows; i++)
    {
        MAT_AT(maxes, i, 0) = MAT_AT(inputs, i, 0);
        for (size_t j = 1; j < inputs.cols; j++)
        {
            if (MAT_AT(inputs, i, j) > MAT_AT(maxes, i, 0))
            {
                MAT_AT(maxes, i, 0) = MAT_AT(inputs, i, j);
            }
        }
    }

    for (size_t i = 0; i < inputs.rows; i++)
    {
        for (size_t j = 0; j < inputs.cols; j++)
        {
            MAT_AT(exp_vals, i, j) = exp(MAT_AT(inputs, i, j) - MAT_AT(maxes, i, 0));
            MAT_AT(exp_sum, i, 0) += MAT_AT(exp_vals, i, j);
        }
        for (size_t k = 0; k < inputs.cols; k++)
        {
            MAT_AT(output, i, k) = MAT_AT(exp_vals, i, k) / MAT_AT(exp_sum, i, 0);
        }
        
    }


}

void Activation_softmax::backward(Mat dvalues)
{
    dinputs = mat_alloc(dvalues.rows,dvalues.cols);

    for (size_t i = 0; i < dvalues.rows; i++)
    {
        
    }
    
}



class ASLCC
{
    ASLCC();
    void forward(Mat inputs, Mat y_true);
    void backward(Mat dvalues, Mat y_true);
    void calculate(Mat Output, Mat y_true);


};


ASLCC::ASLCC()
{
    Activation_softmax activation;

}






class BinaryCrossentropy_Loss
{
public:
    void forward(Mat y_pred, Mat y_true);
    void backward(Mat dvalues, Mat y_true);

    Mat sample_losses;
    Mat y_pred_clipped;
    Mat clipped_dvalues;
    Mat dinputs;
};


void BinaryCrossentropy_Loss::forward(Mat y_pred, Mat y_true)
{
    y_pred_clipped = mat_alloc(y_pred.rows, y_pred.cols);
    sample_losses = mat_alloc(y_pred.rows, y_pred.cols);
    for (size_t i = 0; i < y_pred.rows; i++)
    {
        
        MAT_AT(y_pred_clipped,i,0) = std::clamp(MAT_AT(y_pred,i,0),0.00001f, 0.9999f);
        MAT_AT(sample_losses, i, 0) = -1*((MAT_AT(y_true, i, 0)*std::log(MAT_AT(y_pred_clipped, i, 0))) + ((1 - MAT_AT(y_true, i, 0))*std::log(1 - MAT_AT(y_pred_clipped,i,0))));
    }
}

void BinaryCrossentropy_Loss::backward(Mat dvalues, Mat y_true)
{
    clipped_dvalues = mat_alloc(dvalues.rows, dvalues.cols);
    dinputs = mat_alloc(dvalues.rows, dvalues.cols);
    for (size_t i = 0; i < dvalues.rows; i++)
    {
        MAT_AT(clipped_dvalues, i,0) = std::clamp(MAT_AT(dvalues,i,0),0.00001f, 0.9999f);
        MAT_AT(dinputs,i,0) = -(MAT_AT(y_true,i,0)/MAT_AT(clipped_dvalues,i,0) - (1-MAT_AT(y_true,i,0))/(1-MAT_AT(clipped_dvalues,i,0)));
        MAT_AT(dinputs,i,0) /= dvalues.rows;

    }
    
}

class Optimizer_SGD
{
public:
    Optimizer_SGD(float learning_rate=0.01){  lr = learning_rate;}
    void update_params(LayerDense Layer);

private:
    float lr;
};

void Optimizer_SGD::update_params(LayerDense layer)
{
    for (int i = 0; i < layer.weights.rows; i++)
    {
        for (int j = 0; j < layer.weights.cols; j++)
        {
            MAT_AT(layer.weights, i, j) -= lr* MAT_AT(layer.dweights, i, j);
        }
        MAT_AT(layer.biases, i, 0) -= lr * MAT_AT(layer.dbiases, i, 0);
    }
    
}