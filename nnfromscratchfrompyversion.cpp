
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
    Mat layerinputs;
    Mat layerinputs_T;
};
#endif

LayerDense::LayerDense(int n_inputs, int n_out, bool bias, int num_samples)
{
    n_samples = num_samples;
    layerbias = bias;
    weights = mat_alloc(n_inputs, n_out);
    mat_rand(weights, -0.1, 0.1);
    weights_T = mat_alloc(n_out, n_inputs);
    if (layerbias)
    {
        biases = mat_alloc(1,n_out);
        mat_rand(biases, -1.0,1.0);
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
    Relu_Activation(int num_samples, int num_neurons);
    void forward(Mat inputs);
    void backward(Mat dvalues);

    Mat output;
    Mat layerinputs;
    Mat dinputs;
};

Relu_Activation::Relu_Activation(int num_samples, int num_neurons)
{
    dinputs = mat_alloc(num_samples, num_neurons);
    output = mat_alloc(num_samples, num_neurons);
    layerinputs = mat_alloc(num_samples, num_neurons);
}

void Relu_Activation::forward(Mat inputs)
{
    // layerinputs = mat_alloc(inputs.rows,inputs.cols);
    // output = mat_alloc(inputs.rows, inputs.cols);
    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            MAT_AT(output,i,j) = std::max(0.0,MAT_AT(inputs,i,j));
            MAT_AT(layerinputs,i,j) = MAT_AT(inputs,i,j);
        } 
    }
}


void Relu_Activation::backward(Mat dvalues)
{
    // dinputs = mat_alloc(dvalues.rows, dvalues.cols);
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            MAT_AT(dinputs,i,j) = MAT_AT(dvalues,i,j);
            if (MAT_AT(layerinputs,i,j) <=0.0)
            {
                MAT_AT(dinputs,i,j) = 0.0;
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
    Activation_softmax(int num_samples, int num_classes, int  num_neurons);
    void forward(Mat inputs);
    void backward(Mat dvalues);

    std::vector<Mat> jacobians;
    Mat diagflat;
    Mat output;
    Mat dinputs;
    Mat temp;
    Mat temp_t;
    Mat singleoutput;
    Mat singleoutput_T;
    Mat soutdot;

    Mat exp_vals;
    Mat exp_sum;
    Mat maxes;

     Mat dval_T;
    Mat tempjac;
};



Activation_softmax::Activation_softmax(int num_samples, int num_classes, int  num_neurons)
{
    dinputs = mat_alloc(num_samples,num_classes);
    diagflat = mat_alloc(num_classes,num_classes);

    mat_fill(diagflat, 0.0);
    temp = mat_alloc(num_classes,1);
    temp_t = mat_alloc(1,num_classes);

    exp_vals = mat_alloc(num_samples, num_neurons);
    exp_sum = mat_alloc(num_samples, 1);
    maxes = mat_alloc(num_samples, 1);
    output = mat_alloc(num_samples,  num_classes);
    singleoutput = mat_alloc(1, num_classes);
    singleoutput_T = mat_alloc(num_classes, 1);
    soutdot = mat_alloc(num_classes, num_classes);
   dval_T = mat_alloc(num_classes,1);
   tempjac = mat_alloc(num_classes, num_classes);
   for (size_t i = 0; i < num_samples; i++)
   {
        jacobians.push_back(tempjac);
   }
   
}


void Activation_softmax::forward(Mat inputs)
{   
    mat_fill(exp_sum, 0.0);
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
            MAT_AT(exp_vals, i, j) =  std::exp(MAT_AT(inputs, i, j) - MAT_AT(maxes, i, 0));
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
    //dvalues is 1 x numsamples
    for (size_t i = 0; i < output.rows; i++)
    {   
        singleoutput = mat_row(output, i);
        mat_transpose(singleoutput_T, singleoutput);
        mat_dot(soutdot, singleoutput_T, singleoutput);
        for (size_t j = 0; j < output.cols; j++)
        {
            for (size_t k = 0; k < output.cols; k++)
            {
                if (j == k) 
                {
                    MAT_AT(diagflat, j, k) = MAT_AT(singleoutput, 0,j);
                }
                
                MAT_AT(jacobians[i], j, k) = MAT_AT(diagflat, j, k)  - MAT_AT(soutdot, j, k);
            }
        }
        
        mat_transpose(dval_T, mat_row(dvalues, i));
        //dvalt shape  is numsamples x 1
        mat_dot(temp,  jacobians[i],  dval_T);
        mat_transpose(temp_t, temp);
        for (size_t k = 0; k < dinputs.cols; k++)
        {
            MAT_AT(mat_row(dinputs,i),0,k) = MAT_AT(temp_t,0,k);
        }
    }
    
}



class Loss
{

};


class Loss_categoricalCrossentropy
{
public:
    Loss_categoricalCrossentropy(int num_classes, int num_samples);
    void forward(Mat y_pred, Mat y_true);
    void backward(Mat dvalues, Mat y_true);

    Mat dinputs;
    Mat negative_log_likelihood;

    Mat y_pred_clipped;
    Mat correct_confidences;
};



Loss_categoricalCrossentropy::Loss_categoricalCrossentropy(int num_classes, int num_samples)
{

    y_pred_clipped = mat_alloc(num_classes, num_classes);
    correct_confidences = mat_alloc(num_samples, 1);
    dinputs = mat_alloc(num_samples, num_classes);
    negative_log_likelihood = mat_alloc(1, num_samples);
}



void Loss_categoricalCrossentropy:: forward(Mat y_pred, Mat y_true)
{
    
    for (size_t i = 0; i < y_pred.rows; i++)
    {
        float sum = 0.f;
        for (size_t j = 0; j < y_pred.cols; j++)
        {
            MAT_AT(y_pred_clipped,i,j) = std::clamp(MAT_AT(y_pred, i, j), 1e-7, 1 - 1e-7);
            sum += MAT_AT(y_pred_clipped,i,j) * MAT_AT(y_true, i,j);
        }
        MAT_AT(correct_confidences, i,0) = sum;
        MAT_AT(negative_log_likelihood,0,i) = -std::log(MAT_AT(correct_confidences, i,0));
    }
}

void Loss_categoricalCrossentropy::backward(Mat dvalues, Mat y_true)
{
    int samples = dvalues.rows;
    int labels = dvalues.cols;
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            MAT_AT(dinputs, i,j) = (-MAT_AT(y_true, i,j) / MAT_AT(dvalues,i,j))/(float) samples;
        }   
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
        
        MAT_AT(y_pred_clipped,i,0) = std::clamp(MAT_AT(y_pred,i,0),0.00001, 0.9999);
        MAT_AT(sample_losses, i, 0) = -1*((MAT_AT(y_true, i, 0)*std::log(MAT_AT(y_pred_clipped, i, 0))) + ((1 - MAT_AT(y_true, i, 0))*std::log(1 - MAT_AT(y_pred_clipped,i,0))));
    }
}

void BinaryCrossentropy_Loss::backward(Mat dvalues, Mat y_true)
{
    clipped_dvalues = mat_alloc(dvalues.rows, dvalues.cols);
    dinputs = mat_alloc(dvalues.rows, dvalues.cols);
    for (size_t i = 0; i < dvalues.rows; i++)
    {
        MAT_AT(clipped_dvalues, i,0) = std::clamp(MAT_AT(dvalues,i,0),0.00001, 0.9999);
        MAT_AT(dinputs,i,0) = -(MAT_AT(y_true,i,0)/MAT_AT(clipped_dvalues,i,0) - (1-MAT_AT(y_true,i,0))/(1-MAT_AT(clipped_dvalues,i,0)));
        MAT_AT(dinputs,i,0) /= dvalues.rows;

    }
    
}

class Optimizer_SGD
{
public:
    Optimizer_SGD(double learning_rate=0.01){  lr = learning_rate;}
    void update_params(LayerDense Layer);

private:
    double lr;
};

void Optimizer_SGD::update_params(LayerDense layer)
{
    for (int i = 0; i < layer.weights.rows; i++)
    {
        for (int j = 0; j < layer.weights.cols; j++)
        {
            MAT_AT(layer.weights, i, j) -= lr* MAT_AT(layer.dweights, i, j);
        }
        // if (layer.layerbias)
        // {
        //     // MAT_AT(layer.biases, 0, i) -= lr * MAT_AT(layer.dbiases, 0, i);
        // }
    }

    if (layer.layerbias)
    {
        for (int i = 0; i < layer.biases.cols; i++)
        {
            MAT_AT(layer.biases, 0, i) -= lr * MAT_AT(layer.dbiases, 0, i);
        }
    }
    
}