
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
    void randomize(float percentrand);
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



void LayerDense::randomize(float percentrand)
{
    for (size_t i = 0; i < weights.rows; i++)
    {
        for (size_t j = 0; j < weights.cols; j++)
        {
            float randfl = rand() / RAND_MAX;
            if (randfl > percentrand)
            {
                MAT_AT(weights,i,j) = rand() / RAND_MAX;
            }
            
        }
        
    }
    
}




class Convolution2D
{
public:
    Convolution2D(int depth, int kernel_size, int input_depth, int input_height, int input_width); 
    void forward(std::vector<Mat> inputs);
    void backward(std::vector<std::vector<Mat>> dvalues);
    void cross_correlation2d(Mat a, Mat b, Mat output);
    void convolve2d(Mat a, Mat b, Mat output);

    std::vector<std::vector<Mat>> output;
    std::vector<Mat> inputscopy;
    std::vector<Mat> dinputs;
    std::vector<Mat> biases;
    std::vector<Mat> dbiases;
    std::vector<Mat> weights;
    std::vector<Mat> dweights;
    
};

Convolution2D::Convolution2D(int depth, int kernel_size, int input_depth, int input_height, int input_width)
{
    
    std::vector<Mat> tempout;
    
    for (size_t i = 0; i < input_depth; i++)
    {
        for (size_t j = 0; j < depth; j++)
        {
            tempout.push_back(mat_alloc(input_height - kernel_size + 1, input_width - kernel_size + 1));
        }
        output.push_back(std::move(tempout));
        dinputs.push_back(mat_alloc(input_height, input_width));
        inputscopy.push_back(mat_alloc(input_height, input_width));
    }
    
    for (size_t i = 0; i < depth; i++)
    {
        weights.push_back(mat_alloc(kernel_size, kernel_size));
        mat_rand(weights[i], -1.f, 1.f);
        dweights.push_back(mat_alloc(kernel_size, kernel_size));
        biases.push_back(mat_alloc(input_height-kernel_size+1, input_width-kernel_size+1));
        dbiases.push_back(mat_alloc(input_height-kernel_size+1, input_width-kernel_size+1));
    }
}

void Convolution2D::forward(std::vector<Mat> inputs)
{
    // inputs is input image
    int kernel_size = weights[0].cols;
    int input_height = inputs[0].rows;
    int input_width = inputs[0].cols;
   
    for (size_t i = 0; i < output.size(); i++)
    {
        mat_copy(inputscopy[i], inputs[i]);
        for (size_t j = 0; j < output[i].size(); j++)
        {
            mat_copy(output[i][j], biases[j]);
        }
    }

    for (size_t i = 0; i < inputs.size(); i++)
    {
        for (size_t j = 0; j < output[0].size(); j++)
        {
            cross_correlation2d(inputs[i], weights[j], output[i][j]);
        }
    }
}

void Convolution2D::cross_correlation2d(Mat a, Mat b, Mat output)
{
    // a is input, b is kernel
    int kernel_size = b.rows;
    int input_height = a.rows;
    int input_width = a.cols;
    // std::cout << "dweights " << output.rows << " input h " << input_height << " dvals h " << kernel_size << std::endl;
    for (size_t i = 0; i < input_height - kernel_size+1; i++)
    {
        for (size_t j = 0; j < input_width - kernel_size+1; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < kernel_size; k++)
            {
                for (size_t l = 0; l < kernel_size; l++)
                {
                    int ii = i - k;
                    int jj = j - l;
                    if (ii >= 0 && ii < input_height && jj >= 0 && jj < input_width)
                    {
                        sum += MAT_AT(a, ii, jj) * MAT_AT(b, k, l);
                    }
                    
                }   
            }
            MAT_AT(output, i, j) += sum;
        }
    }
}

//this is wrong.
void Convolution2D::convolve2d(Mat a, Mat b, Mat output)
{
    // a is input, b is kernel
    int kernel_height = b.rows;
    int kernel_width = b.cols;
    int input_height = a.rows;
    int input_width = a.cols;

    for (size_t i = 0; i < input_height; i++)
    {
        // std::cout << i << " conv\n";
        for (size_t j = 0; j < input_width; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < kernel_height; k++)
            {
                for (size_t l = 0; l < kernel_width; l++)
                {
                    int ii = i - k;
                    int jj = j - l;
                    if (ii >=0 && ii < kernel_height && jj >=  0 && jj < kernel_width)
                    {
                        sum = MAT_AT(a, ii, jj) * MAT_AT(b, b.rows-1-k, b.cols-1-l );
                    }
                }
            }
            MAT_AT(output, i, j) = sum;
        }
    }
}

void Convolution2D::backward(std::vector<std::vector<Mat>> dvalues)
{
    
    for (size_t i = 0; i < dweights.size(); i++)
    {
        mat_fill(dweights[i], 0.f);
    }
    
    for (size_t i = 0; i < output.size(); i++)
    {
        mat_fill(dinputs[i], 0.f);
        // std::cout << "here " << dweights[i].rows << std::endl;
        for (size_t j = 0; j < output[0].size(); j++)
        {
            //fix this.../././././.
            // mat_copy(dbiases[i][j], dvalues[i][j]);
            for (size_t k = 0; k < dvalues[i][j].rows; k++)
            {
                for (size_t l = 0; l < dvalues[i][j].cols; l++)
                {
                    MAT_AT(dbiases[j],k,l) += MAT_AT(dvalues[i][j],k,l);
                }   
            }
            cross_correlation2d(inputscopy[i], dvalues[i][j], dweights[j]);
            convolve2d(dvalues[i][j], weights[j], dinputs[i]);
        }
    }
    
}

class Relu_Activation
{
public:

    Relu_Activation(int num_samples, int num_neurons);
    Relu_Activation(int num_samples, int num_neurons, int numdvalrows, int numdvalcols);
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

Relu_Activation::Relu_Activation(int num_samples, int num_neurons, int numdvalrows, int numdvalcols)
{
    dinputs = mat_alloc(num_samples, num_neurons);
    output = mat_alloc(num_samples, num_neurons);
    layerinputs = mat_alloc(num_samples, num_neurons);
    dinputs = mat_alloc(numdvalrows, numdvalcols);
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
//    tempjac = mat_alloc(num_classes, num_classes);
   for (size_t i = 0; i < num_samples; i++)
   {
        jacobians.push_back(mat_alloc(num_classes, num_classes));
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
    void update_convparams(Convolution2D conv);

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
    }

    if (layer.layerbias)
    {
        for (int i = 0; i < layer.biases.cols; i++)
        {
            MAT_AT(layer.biases, 0, i) -= lr * MAT_AT(layer.dbiases, 0, i);
        }
    }
    
}

void Optimizer_SGD::update_convparams(Convolution2D conv)
{
    int k = 0;
    for (const Mat kernel : conv.weights)
    {
        for (int i = 0; i < kernel.rows; i++)
        {
            for (int j = 0; j < kernel.cols; j++)
            {
                MAT_AT(kernel, i, j) -= lr* MAT_AT(conv.dweights[k], i, j);
            }
        }
        k++;
    }
    k = 0;
    for (const auto lbiases: conv.biases)
        {
            for (size_t i = 0; i < lbiases.rows; i++)
            {
                for (size_t j = 0; j < lbiases.cols; j++)
                {
                    MAT_AT(lbiases, 0, i) -= lr * MAT_AT(conv.dbiases[k], 0, i);
                }
                
            }
            k++;    
        
        }
    
}





//claude self attention layer
class SelfAttentionLayer
{
public:
    SelfAttentionLayer(int n_inputs, int d_model, int num_heads, int seq_len, int num_samples);
    void forward(Mat inputs);
    void backward(Mat dvalues);
    
    Mat output;
    Mat dinputs;
    Mat dweights_q;
    Mat dweights_k; 
    Mat dweights_v;
    Mat dweights_o;
    Mat dbiases_q;
    Mat dbiases_k;
    Mat dbiases_v;
    Mat dbiases_o;

//private:
    Mat weights_q;
    Mat weights_k;
    Mat weights_v;
    Mat weights_o;
    Mat biases_q;
    Mat biases_k;
    Mat biases_v;
    Mat biases_o;
    
    // Internal computation matrices
    Mat Q, K, V;
    Mat attention_scores;
    Mat attention_weights;
    Mat attention_output;
    Mat layerinputs;
    
    // Multi-head matrices
    std::vector<Mat> Q_heads;
    std::vector<Mat> K_heads;
    std::vector<Mat> V_heads;
    std::vector<Mat> attention_heads;
    std::vector<Mat> score_heads;
    std::vector<Mat> weight_heads;
    
    int d_model;
    int num_heads;
    int d_k; // d_model / num_heads
    int seq_len;
    int n_samples;
    bool layerbias;
    
private:
    void compute_attention_head(int head_idx);
    void softmax_2d(Mat& matrix);
    void reshape_for_heads();
    void concatenate_heads();
};

SelfAttentionLayer::SelfAttentionLayer(int n_inputs, int d_model_size, int num_heads_count, int sequence_len, int num_samples)
{
    d_model = d_model_size;
    num_heads = num_heads_count;
    d_k = d_model / num_heads;
    seq_len = sequence_len;
    n_samples = num_samples;
    layerbias = true;
    
    // Allocate weight matrices
    weights_q = mat_alloc(n_inputs, d_model);
    weights_k = mat_alloc(n_inputs, d_model);
    weights_v = mat_alloc(n_inputs, d_model);
    weights_o = mat_alloc(d_model, d_model);
    
    // Initialize weights randomly
    mat_rand(weights_q, -0.1, 0.1);
    mat_rand(weights_k, -0.1, 0.1);
    mat_rand(weights_v, -0.1, 0.1);
    mat_rand(weights_o, -0.1, 0.1);
    
    // Allocate bias vectors
    if (layerbias)
    {
        biases_q = mat_alloc(1, d_model);
        biases_k = mat_alloc(1, d_model);
        biases_v = mat_alloc(1, d_model);
        biases_o = mat_alloc(1, d_model);
        
        // Initialize biases
        mat_fill(biases_q, 0.0);
        mat_fill(biases_k, 0.0);
        mat_fill(biases_v, 0.0);
        mat_fill(biases_o, 0.0);
        
        // Gradient matrices for biases
        dbiases_q = mat_alloc(1, d_model);
        dbiases_k = mat_alloc(1, d_model);
        dbiases_v = mat_alloc(1, d_model);
        dbiases_o = mat_alloc(1, d_model);
    }
    
    // Allocate gradient matrices for weights
    dweights_q = mat_alloc(n_inputs, d_model);
    dweights_k = mat_alloc(n_inputs, d_model);
    dweights_v = mat_alloc(n_inputs, d_model);
    dweights_o = mat_alloc(d_model, d_model);
    
    // Allocate intermediate computation matrices
    Q = mat_alloc(n_samples * seq_len, d_model);
    K = mat_alloc(n_samples * seq_len, d_model);
    V = mat_alloc(n_samples * seq_len, d_model);
    attention_scores = mat_alloc(n_samples * seq_len, seq_len);
    attention_weights = mat_alloc(n_samples * seq_len, seq_len);
    attention_output = mat_alloc(n_samples * seq_len, d_model);
    
    // Allocate input/output matrices
    layerinputs = mat_alloc(n_samples * seq_len, n_inputs);
    output = mat_alloc(n_samples * seq_len, d_model);
    dinputs = mat_alloc(n_samples * seq_len, n_inputs);
    
    // Allocate multi-head matrices
    for (int i = 0; i < num_heads; i++)
    {
        Q_heads.push_back(mat_alloc(n_samples * seq_len, d_k));
        K_heads.push_back(mat_alloc(n_samples * seq_len, d_k));
        V_heads.push_back(mat_alloc(n_samples * seq_len, d_k));
        attention_heads.push_back(mat_alloc(n_samples * seq_len, d_k));
        score_heads.push_back(mat_alloc(n_samples * seq_len, seq_len));
        weight_heads.push_back(mat_alloc(n_samples * seq_len, seq_len));
    }
}

void SelfAttentionLayer::forward(Mat inputs)
{
    // Store inputs for backward pass
    mat_copy(layerinputs, inputs);
    
    // Compute Q, K, V projections
    if (layerbias)
    {
        mat_dot_bias(Q, inputs, weights_q, biases_q);
        mat_dot_bias(K, inputs, weights_k, biases_k);
        mat_dot_bias(V, inputs, weights_v, biases_v);
    }
    else
    {
        mat_dot(Q, inputs, weights_q);
        mat_dot(K, inputs, weights_k);
        mat_dot(V, inputs, weights_v);
    }
    // Reshape Q, K, V for multi-head attention
    reshape_for_heads();
    
    // Compute attention for each head
    for (int h = 0; h < num_heads; h++)
    {
        compute_attention_head(h);
    }
    
    // // Concatenate heads
    // concatenate_heads();
    
    // Apply output projection
    // if (layerbias)
    // {
    //     mat_dot_bias(output, attention_output, weights_o, biases_o);
    // }
    // else
    // {
    //     mat_dot(output, attention_output, weights_o);
    // }
}

void SelfAttentionLayer::compute_attention_head(int head_idx)
{
    // Compute attention scores: Q * K^T
    Mat K_head_T = mat_alloc(d_k, n_samples * seq_len);
    mat_transpose(K_head_T, K_heads[head_idx]);
    mat_dot(score_heads[head_idx], Q_heads[head_idx], K_head_T);
    
    // Scale by sqrt(d_k)
    double scale_factor = 1.0 / std::sqrt((double)d_k);
    for (size_t i = 0; i < score_heads[head_idx].rows; i++)
    {
        for (size_t j = 0; j < score_heads[head_idx].cols; j++)
        {
            MAT_AT(score_heads[head_idx], i, j) *= scale_factor;
        }
    }
    
    // Apply softmax to get attention weights
    mat_copy(weight_heads[head_idx], score_heads[head_idx]);
    softmax_2d(weight_heads[head_idx]);
    
    // Apply attention weights to values: Attention * V
    mat_dot(attention_heads[head_idx], weight_heads[head_idx], V_heads[head_idx]);
    
    mat_free(K_head_T);
}

void SelfAttentionLayer::softmax_2d(Mat& matrix)
{
    // Apply softmax to each row (each sequence position attends to all positions)
    for (size_t i = 0; i < matrix.rows; i++)
    {
        // Find max for numerical stability
        double max_val = MAT_AT(matrix, i, 0);
        for (size_t j = 1; j < matrix.cols; j++)
        {
            if (MAT_AT(matrix, i, j) > max_val)
            {
                max_val = MAT_AT(matrix, i, j);
            }
        }
        
        // Compute exp and sum
        double sum = 0.0;
        for (size_t j = 0; j < matrix.cols; j++)
        {
            MAT_AT(matrix, i, j) = std::exp(MAT_AT(matrix, i, j) - max_val);
            sum += MAT_AT(matrix, i, j);
        }
        
        // Normalize
        for (size_t j = 0; j < matrix.cols; j++)
        {
            MAT_AT(matrix, i, j) /= sum;
        }
    }
}

void SelfAttentionLayer::reshape_for_heads()
{
    // Split Q, K, V into multiple heads
    for (int h = 0; h < num_heads; h++)
    {
        for (size_t i = 0; i < Q.rows; i++)
        {
            for (int j = 0; j < d_k; j++)
            {
                MAT_AT(Q_heads[h], i, j) = MAT_AT(Q, i, h * d_k + j);
                MAT_AT(K_heads[h], i, j) = MAT_AT(K, i, h * d_k + j);
                MAT_AT(V_heads[h], i, j) = MAT_AT(V, i, h * d_k + j);
            }
        }
    }
}

void SelfAttentionLayer::concatenate_heads()
{
    // Concatenate attention heads back together
    for (int h = 0; h < num_heads; h++)
    {
        for (size_t i = 0; i < attention_heads[h].rows; i++)
        {
            for (int j = 0; j < d_k; j++)
            {
                MAT_AT(attention_output, i, h * d_k + j) = MAT_AT(attention_heads[h], i, j);
            }
        }
    }
}

void SelfAttentionLayer::backward(Mat dvalues)
{
    // This is a simplified backward pass - full implementation would require
    // computing gradients through the attention mechanism
    
    // Gradient w.r.t. output projection
    Mat layerinputs_T = mat_alloc(layerinputs.cols, layerinputs.rows);
    mat_transpose(layerinputs_T, layerinputs);
    
    // Compute weight gradients (simplified - should use attention_output)
    mat_dot(dweights_o, layerinputs_T, dvalues);
    
    // Compute bias gradients
    if (layerbias)
    {
        for (size_t j = 0; j < dbiases_o.cols; j++)
        {
            MAT_AT(dbiases_o, 0, j) = 0.0;
            for (size_t i = 0; i < dvalues.rows; i++)
            {
                MAT_AT(dbiases_o, 0, j) += MAT_AT(dvalues, i, j);
            }
        }
    }
    
    // Gradient w.r.t. inputs (simplified)
    Mat weights_o_T = mat_alloc(weights_o.cols, weights_o.rows);
    mat_transpose(weights_o_T, weights_o);
    mat_dot(dinputs, dvalues, weights_o_T);
    
    // Note: Full backward pass would require:
    // 1. Backprop through attention weights (softmax gradient)
    // 2. Backprop through scaled dot-product attention
    // 3. Backprop through Q, K, V projections
    // This is simplified for brevity
    
    mat_free(layerinputs_T);
    mat_free(weights_o_T);
}