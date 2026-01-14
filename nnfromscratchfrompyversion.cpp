
#ifndef NN_H
#define NN_H
#include <algorithm>
#include "tensorstuff.cpp"
#include  <random>
#include <chrono>
#include <map>

class LayerDense
{
public:
    LayerDense(int n_inputs, int n_out, bool bias, int num_samples);
    LayerDense(int embed_dim, int n_out, bool bias, int batch_size, int seq_len);
    void forward(Mat inputs);
    void forward(Tensor inputs);
    void halfforward(Mat inputs);
    void halfbackward(Mat dvalues);
    void backward(Mat dvalues);
    void backward(Tensor dvalues);
    void randomize(float percentrand);
    //~LayerDense();
    Mat output;
    //for testing half dot learning.
    Mat duboutput;
    //output for tensors forward.
    Tensor toutput;
    Mat dinputs;
    Mat dbiases;

//private:
    Mat weights;
    Mat weights_T;
    Mat biases;
    bool layerbias;
    int n_samples;
    
    Mat dweights;
    // Mat doubleweights;
    Mat layerinputs;
    Mat layerinputs_T;
    Tensor tlayerinputs;
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
    duboutput = mat_alloc(n_samples, 2*n_out);
}

LayerDense::LayerDense(int embed_dim, int n_out, bool bias, int batch_size, int seq_len)
{
    n_samples = embed_dim;
    layerbias = bias;
    weights = mat_alloc(embed_dim, n_out);
    mat_rand(weights, -0.1, 0.1);
    weights_T = mat_alloc(n_out, embed_dim);
    if (layerbias)
    {
        biases = mat_alloc(1,n_out);
        mat_rand(biases, -1.0,1.0);
        dbiases = mat_alloc(1, n_out);
    }
    dweights = mat_alloc(embed_dim, n_out);
    dinputs = mat_alloc(batch_size,embed_dim);
    layerinputs = mat_alloc(batch_size, embed_dim);
    layerinputs_T = mat_alloc(embed_dim, batch_size);
    toutput = tensor_alloc(batch_size, seq_len,n_out);
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

void LayerDense::halfforward(Mat inputs)
{
    layerinputs = inputs;
    //no biases yet.
    mat_half_dot(duboutput, inputs, weights);
}

void LayerDense::forward(Tensor inputs)
{
    tlayerinputs = inputs;
    if (layerbias)
    {
        //bmm(toutput, inputs, weights);
        std::cout <<  "did not make tensor version of dot product with bias in layerdense \n";
    }
    else
    {
        bmm(toutput, inputs, weights);
        
    }
}

void LayerDense::backward(Mat dvalues)
{
    mat_transpose(layerinputs_T,  layerinputs);
    mat_dot(dweights, layerinputs_T, dvalues);
    
    if (layerbias)
    {
        for (size_t i = 0; i < dvalues.cols; i++)
        {
            for (size_t j = 0; j < dvalues.rows; j++)
            {
                MAT_AT(dbiases, 0, i) += MAT_AT(dvalues, j, i);
            }
            
        }
    }
    //...........
    mat_transpose(weights_T, weights);
    mat_dot(dinputs, dvalues, weights_T);
}

void LayerDense::halfbackward(Mat dvalues)
{


   // ----------------------------------------------------------------
    // 1. Calculate Gradients for Weights (dweights)
    // ----------------------------------------------------------------
    // We reverse the forward pass logic:
    // - Top half of weights (rows 0 to N/2) contributed to Left Output (cols 0 to D)
    // - Bottom half (rows N/2 to N) contributed to Right Output (cols D to 2D)

    size_t n = weights.rows;      // N
    size_t d = weights.cols;      // D (half of dvalues.cols)
    size_t r = layerinputs.rows;  // Batch size

    // Reset dweights to 0
    mat_fill(dweights, 0);

    // Part A: Top half of weights gets gradient from Left half of dvalues
    // dW_top = inputs_left^T * dvalues_left
    for (size_t k = 0; k < n / 2; k++) {
        for (size_t j = 0; j < d; j++) {
            float acc = 0.0f;
            for (size_t i = 0; i < r; i++) {
                acc += MAT_AT(layerinputs, i, k) * MAT_AT(dvalues, i, j);
            }
            MAT_AT(dweights, k, j) = acc;
        }
    }

    // Part B: Bottom half of weights gets gradient from Right half of dvalues
    // dW_bottom = inputs_right^T * dvalues_right
    for (size_t k = n / 2; k < n; k++) {
        for (size_t j = 0; j < d; j++) {
            float acc = 0.0f;
            for (size_t i = 0; i < r; i++) {
                // Note: dvalues column is offset by 'd' (the right half)
                acc += MAT_AT(layerinputs, i, k) * MAT_AT(dvalues, i, d + j);
            }
            MAT_AT(dweights, k, j) = acc;
        }
    }

    // ----------------------------------------------------------------
    // 2. Calculate Gradients for Bias (dbiases)
    // ----------------------------------------------------------------
    
    // if (layerbias)
    // {
    //     // Standard summation over batch dimension
    //     for (size_t j = 0; j < dvalues.cols; j++) // Iterate 0 to 2D
    //     {
    //         MAT_AT(dbiases, 0, j) = 0;
    //         for (size_t i = 0; i < dvalues.rows; i++)
    //         {
    //             MAT_AT(dbiases, 0, j) += MAT_AT(dvalues, i, j);
    //         }
    //     }
    // }

    // ----------------------------------------------------------------
    // 3. Calculate Gradients for Inputs (dinputs)
    // ----------------------------------------------------------------
    // dinputs = dvalues * weights_T
    // Split logic:
    // - Left Input Grads (cols 0 to N/2) come from Left dvalues * Top Weights
    // - Right Input Grads (cols N/2 to N) come from Right dvalues * Bottom Weights

    // Reset dinputs to 0
    mat_fill(dinputs, 0);

    // Part A: Left half of inputs (dvalues_left * weights_top_T)
    for (size_t i = 0; i < r; i++) {
        for (size_t k = 0; k < n / 2; k++) {
            float acc = 0.0f;
            for (size_t j = 0; j < d; j++) {
                // weights[k, j] is effectively weights_T[j, k]
                // We use dvalues[i, j] (Left half)
                acc += MAT_AT(dvalues, i, j) * MAT_AT(weights, k, j);
            }
            MAT_AT(dinputs, i, k) = acc;
        }
    }

    // Part B: Right half of inputs (dvalues_right * weights_bottom_T)
    for (size_t i = 0; i < r; i++) {
        for (size_t k = n / 2; k < n; k++) {
            float acc = 0.0f;
            for (size_t j = 0; j < d; j++) {
                // We use dvalues[i, d+j] (Right half)
                // Weights accessed at [k, j] which is correct for bottom block
                acc += MAT_AT(dvalues, i, d + j) * MAT_AT(weights, k, j);
            }
            MAT_AT(dinputs, i, k) = acc;
        }
    }
}


void LayerDense::backward(Tensor dvalues)
{

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


class Subsetlayer
{
public:
    Subsetlayer(int n_inputs, int n_out, bool bias, int num_samples, const std::vector<std::tuple<int, int>> &randgrads);
    // LayerDense(int embed_dim, int n_out, bool bias, int batch_size, int seq_len);
    void forward(Mat inputs);
    void forward(Tensor inputs);
    void halfforward(Mat inputs);
    void halfbackward(Mat dvalues);
    void backward(Mat dvalues);
    void backward(Tensor dvalues);
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
    // Mat doubleweights;
    Mat layerinputs;
    Mat layerinputs_T;
    std::vector<std::tuple<int, int>> fewgrads;
};


Subsetlayer::Subsetlayer(int n_inputs, int n_out, bool bias, int num_samples, const std::vector<std::tuple<int, int>> &randgrads)
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
    fewgrads = randgrads;
    
}

void Subsetlayer::forward(Mat inputs)
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

void Subsetlayer::backward(Mat dvalues)
{
    // mat_transpose(layerinputs_T,  layerinputs);
    // mat_dot(dweights, layerinputs_T, dvalues);
    mat_fill(dweights, 0.0);
    for(const auto& [row, col] : fewgrads)
    {
        for (size_t i = 0; i < layerinputs.rows; i++)
        {
            MAT_AT(dweights, row, col) += MAT_AT(layerinputs, i, row) * MAT_AT(dvalues, i, col);
        }
    }
    



    if (layerbias)
    {
        for (size_t i = 0; i < dvalues.cols; i++)
        {
            for (size_t j = 0; j < dvalues.rows; j++)
            {
                MAT_AT(dbiases, 0, i) += MAT_AT(dvalues, j, i);
            }
            
        }
    }
    //...........
    mat_transpose(weights_T, weights);
    mat_dot(dinputs, dvalues, weights_T);
}



class Lora
{
public:
    Lora(int n_inputs, int n_out, bool bias, int num_samples, int rank);
    void forward(Mat inputs);
    void backward(Mat dvalues);
    void bigBack(Mat dvalues);

    Mat output;
   
    Mat dinputs;
    Mat dbiases;

//private:
    Mat wA, wB;
    // Mat weights_T;
    Mat biases;
    bool layerbias;
    int n_samples, rank;
    
    Mat dwA, dwB;
    Mat layerinputs;
    // Mat layerinputs_T;
};

Lora::Lora(int n_inputs, int n_out, bool bias, int num_samples, int r)
{
    n_samples = num_samples;
    layerbias = bias;
    rank = r;
    wA = mat_alloc(n_inputs, rank);
    wB = mat_alloc(rank, n_out);
    
    mat_rand(wA, -1, 1);
    mat_rand(wB, -0.5, 0.5);
    if (layerbias)
    {
        biases = mat_alloc(1,n_out);
        mat_rand(biases, -1.0,1.0);
        dbiases = mat_alloc(1, n_out);
    }
    dwA = mat_alloc(n_inputs, rank);
    dwB = mat_alloc(rank, n_out);
    dinputs = mat_alloc(num_samples,n_inputs);
    layerinputs = mat_alloc(num_samples, n_inputs);
    // layerinputs_T = mat_alloc(n_inputs, num_samples);
    output = mat_alloc(n_samples,n_out);
}

void Lora::forward(Mat inputs)
{
    assert(inputs.rows == output.rows);
    assert(wB.cols == output.cols);
    assert(wA.cols == wB.rows);
    assert(wA.rows ==  inputs.cols);
    assert(wB.cols == output.cols);
    
    layerinputs = inputs;
    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            //should probably move outside of all for loops if you want faster. 
            if(layerbias) MAT_AT(output,i,j) = MAT_AT(biases, 0,j);
            else MAT_AT(output, i,j) = 0.0;
            for (size_t l = 0; l < inputs.cols; l++)
            {
                double totalweight{0};
                for (size_t k = 0; k < rank; k++)
                {
                    totalweight += MAT_AT(wA, l,k) * MAT_AT(wB, k, j);
                    
                }
                MAT_AT(output, i,j) += MAT_AT(inputs, i, l) * totalweight;
            }
            
        }
       
    }
    
}


void Lora::bigBack(Mat dvalues)
{
    if (layerbias)
    {
        for (size_t i = 0; i < dvalues.cols; i++)
        {
            for (size_t j = 0; j < dvalues.rows; j++)
            {
                MAT_AT(dbiases, 0, i) += MAT_AT(dvalues, j, i);
            }        
        }
    }
    
    Mat inpwa = mat_alloc(layerinputs.rows, wA.cols);
    Mat inpwa_t = mat_alloc(wA.cols, layerinputs.rows);
    Mat layerinputs_t = mat_alloc(layerinputs.cols, layerinputs.rows);
    Mat dvalwb = mat_alloc(dvalues.rows, wB.rows);
    Mat wa_t = mat_alloc(wA.cols, wA.rows);
    Mat wb_t = mat_alloc(wB.cols, wB.rows);

    //calculate dvalwb here
    mat_transpose(wb_t, wB);

    mat_dot(dvalwb, dvalues, wb_t);


    //inputs and wa combined
    mat_dot(inpwa, layerinputs, wA);
   
    // dot of inputwa and dvals to get dwb
    mat_transpose(inpwa_t, inpwa);
    mat_dot(dwB, inpwa_t, dvalues);    
    //inputs transposed dotted with mix of dvalues and wb 

    mat_transpose(layerinputs_t, layerinputs);
    mat_dot(dwA, layerinputs_t, dvalwb);

    //calculate dinputs, don't need since first layer
    mat_dot(dinputs, dvalwb, wa_t);



    mat_free(inpwa);
    mat_free(inpwa_t);
    mat_free(layerinputs_t);
    mat_free(dvalwb);
    mat_free(wa_t);
    mat_free(wb_t);
   
}

void Lora::backward(Mat dvalues)
{
   

    // IMPORTANT: grad_wA and grad_wB must start at 0.0 because we accumulate +=
    mat_fill(dwA, 0);
    mat_fill(dwB, 0);

    //gemini code....... 
    size_t N = layerinputs.rows;      // Batch size
    size_t I = layerinputs.cols;      // Input features
    size_t J = dvalues.cols; // Output features
    size_t R = rank;             // LoRA Rank

    // 1. Bias Gradients
    // Simple accumulation, no extra memory needed
    if (layerbias)
    {
        for (size_t j = 0; j < J; j++)
        {
            double sum = 0.0;
            for (size_t n = 0; n < N; n++) sum += MAT_AT(dvalues, n, j);
            MAT_AT(dbiases, 0, j) = sum;
        }
    }

    // 2. Weights and Input Gradients
    // We iterate through the Batch (N) and Rank (R) first.
    // Inside these loops, we calculate temporary scalars on the fly.
    
    for (size_t n = 0; n < N; n++)
    {
        for (size_t r = 0; r < R; r++)
        {
            // --- Scalar A: Project Gradient backward through B ---
            // Represents (grad_output[n] . wB[r]^T)
            // We compute this single double value by looping over J
            double grad_proj_scalar = 0.0;
            for (size_t j = 0; j < J; j++)
            {
                grad_proj_scalar += MAT_AT(dvalues, n, j) * MAT_AT(wB, r, j);
            }
    
            // --- Scalar B: Project Input forward through A ---
            // Represents (inputs[n] . wA[r])
            // We compute this single double value by looping over I
            double input_proj_scalar = 0.0;
            for (size_t i = 0; i < I; i++)
            {
                input_proj_scalar += MAT_AT(layerinputs, n, i) * MAT_AT(wA, i, r);
            }

            // --- Apply to Gradients ---
            // std::cout << "grad proj " << grad_proj_scalar << " ";
            // Update grad_wB (using Scalar B)
            // dL/dwB += (Input * wA)^T * grad_output
            for (size_t j = 0; j < J; j++)
            {
                MAT_AT(dwB, r, j) += input_proj_scalar * MAT_AT(dvalues, n, j);
            }

            // Update grad_wA and grad_inputs (using Scalar A)
            // dL/dwA    += Input^T * (grad_output * wB^T)
            // dL/dInput += (grad_output * wB^T) * wA^T
            for (size_t i = 0; i < I; i++)
            {
                // Update Weight A Gradient
                MAT_AT(dwA, i, r) += MAT_AT(layerinputs, n, i) * grad_proj_scalar;
                // Update Input Gradient (to pass to previous layer)
                MAT_AT(dinputs, n, i) += grad_proj_scalar * MAT_AT(wA, i, r);
            }
        }
        // std::cout << std::endl;
        //...........
    }







    //my code........######
    // //find dwb
    // for (size_t i = 0; i < R; i++)
    // {
    //     for (size_t j = 0; j < outputcols; j++)
    //     {
    //         double inpwa{0};
    //         for (size_t k = 0; k < inputrows; k++)
    //         {
    //             for (size_t l = 0; l < inputcols; l++)
    //             {
    //                 inpwa += MAT_AT(layerinputs, k,l) * MAT_AT(wA, l, i);
    //             }
    //         }
    //         for (size_t k = 0; k < inputrows; k++)
    //         {
    //             MAT_AT(dwB, i,j) += inpwa * MAT_AT(dvalues, k, j);
    //         }
            
    //     }
        
    // }
    
    //##############

}














//seems worthless but maybe not worthless. if trying again, use without relu between this and next layer. possibly don't use any nonlinearity.
class LayerKron
{
public:
    LayerKron(int n_inputs, int n_out, bool bias, int num_samples);
    void forward(Mat inputs);
    void backward(Mat dvalues);


    Mat output;
    Mat output_t;
    Mat dinputs;
    Mat dbiases;

//private:
    Mat weights;
    Mat weights_avg;
    Mat biases;
    bool layerbias;
    int n_samples;
    
    Mat dweights;
    Mat layerinputs;
    Mat layerinputs_avg;
    Mat dval_avg;
    Mat dvals_t;
    
};

LayerKron::LayerKron(int n_inputs, int n_out, bool bias, int num_samples)
{
    n_samples = num_samples;
    layerbias = bias;
    weights = mat_alloc(n_inputs, n_out);
    mat_rand(weights, -0.1, 0.1);
    if (layerbias)
    {
        //not sure how i want this yet. for now same as  output_t
        biases = mat_alloc(n_inputs*n_out,num_samples);
        mat_rand(biases, -1.0,1.0);
        dbiases = mat_alloc(n_inputs*n_out,num_samples);
    }
    dweights = mat_alloc(n_inputs, n_out);
    dinputs = mat_alloc(num_samples,n_inputs);
    layerinputs = mat_alloc(n_samples, n_inputs);
    layerinputs_avg = mat_alloc(1,n_inputs);
    dval_avg = mat_alloc(1,n_inputs*n_out);
    weights_avg = mat_alloc(1, n_inputs);
    //a_cols x b_cols, a_rows
    output_t = mat_alloc(n_inputs*n_out,num_samples);
    //need to transpose to plug into next layer
    output = mat_alloc(n_samples, n_inputs*n_out);
    dvals_t = mat_alloc(n_inputs*n_out, num_samples); 
}


void LayerKron::forward(Mat inputs)
{
    layerinputs = inputs;
    if (layerbias) { rowtocolkron_bias(output_t, inputs, weights, biases); }
    else { rowtocolkron(output_t, inputs, weights); }
    mat_transpose(output, output_t);
}

//unfinished but I think not necessary to finish if you keep it as the first layer, only missing deriv of inputs to layer. 
//if trying again, use without relu between this and next layer. possibly don't use any nonlinearity.
void LayerKron::backward(Mat dvalues)
{
    mat_fill(layerinputs_avg, 0);
    //ok first get average of input along dim=1. then do the same with dvals. then mult inputavg by dvaluesavg will give gradient of weights.
    for (size_t i = 0; i < layerinputs.rows; i++)
    {
        for (size_t j = 0; j < layerinputs.cols; j++)
        {
            MAT_AT(layerinputs_avg, 0,j) += (MAT_AT(layerinputs, i, j) / (double)layerinputs.cols);
        }
    }
    // mat_print(layerinputs_avg);
    for (size_t i = 0; i < dvalues.rows; i++)
    {
        for (size_t j = 0; j < dvalues.cols; j++)
        {
            MAT_AT(dval_avg, 0, j) += MAT_AT(dvalues, i,j) / (double)dvalues.cols;
        }
    }
    
    //also get weights avg while we are here for getting dinputs later.
    for (size_t i = 0; i < weights.rows; i++)
    {
        for (size_t j = 0; j < weights.cols; j++)
        {
            MAT_AT(weights_avg, 0, i) += MAT_AT(weights, i, j) / (double)weights.rows;
        }
    }

    //these two parts are a problem........

    //next get grads from dvals and inputs
    mat_fill(dweights, 0);
    for (size_t i = 0; i < dweights.rows; i++)
    {
        for (size_t j = 0; j < dweights.cols; j++)
        {
            MAT_AT(dweights, i, j) = MAT_AT(dval_avg, 0,i*dweights.cols+j) * MAT_AT(layerinputs_avg, 0, i);
        }
    }
    
    
    if (layerbias)
    {
       for (size_t i = 0; i < dbiases.rows; i++)
       {
           for (size_t j = 0; j < dbiases.cols; j++)
           {
            //do dvals j, i indexing because i had to flip the output to get numsamples by number of neurons for the next layer.
                MAT_AT(dbiases, i, j) = MAT_AT(dvalues, j, i);
           }   
       }
    }
   
    //next find the dinputs

}


//I think this just maps to a new number so it is essentially useless. 
// class SqLayerDense
// {
// public:
//     //for mats
//     SqLayerDense(int n_inputs, int n_out, bool bias, int num_samples);
//     //for tensors
//     SqLayerDense(int embed_dim, int n_out, bool bias, int batch_size, int seq_len);
//     void forward(Mat inputs);
//     void forward(Tensor inputs);
//     void backward(Mat dvalues);
//     void backward(Tensor dvalues);
//     void randomize(float percentrand);
//     //~LayerDense();
//     Mat output;
//     Tensor toutput;
//     Mat dinputs;
//     Mat dbiases;

// //private:
//     Mat weights;
//     Mat weights_T;
//     Mat biases;
//     bool layerbias;
//     int n_samples;
    
//     Mat dweights;
//     Mat layerinputs;
//     Mat layerinputs_T;
//     Tensor tlayerinputs;

//     Mat doubleweights;
//     Mat weightsTsq;
//     float pow;
// };

// SqLayerDense::SqLayerDense(int n_inputs, int n_out, bool bias, int num_samples)
// {
//     n_samples = num_samples;
//     pow = 2;
//     layerbias = bias;
//     weights = mat_alloc(n_inputs, n_out);
//     mat_rand(weights, -0.1, 0.1);
//     weights_T = mat_alloc(n_out, n_inputs);
//     if (layerbias)
//     {
//         biases = mat_alloc(1,n_out);
//         mat_rand(biases, -1.0,1.0);
//         dbiases = mat_alloc(1, n_out);
//     }
//     dweights = mat_alloc(n_inputs, n_out);
//     doubleweights = mat_alloc(n_inputs, n_out);
//     dinputs = mat_alloc(num_samples,n_inputs);
//     layerinputs = mat_alloc(num_samples, n_inputs);
//     layerinputs_T = mat_alloc(n_inputs, num_samples);
//     output = mat_alloc(n_samples,n_out);
//     weightsTsq = mat_alloc(n_out, n_inputs);
// }

// SqLayerDense::SqLayerDense(int embed_dim, int n_out, bool bias, int batch_size, int seq_len)
// {
//     n_samples = embed_dim;
//     pow = 2;
//     layerbias = bias;
//     weights = mat_alloc(embed_dim, n_out);
//     mat_rand(weights, -0.1, 0.1);
//     weights_T = mat_alloc(n_out, embed_dim);
//     // if (layerbias)
//     // {
//     //     biases = mat_alloc(1,n_out);
//     //     mat_rand(biases, -1.0,1.0);
//     //     dbiases = mat_alloc(1, n_out);
//     // }
//     dweights = mat_alloc(embed_dim, n_out);
//     dinputs = mat_alloc(batch_size,embed_dim);
//     layerinputs = mat_alloc(batch_size, embed_dim);
//     layerinputs_T = mat_alloc(embed_dim, batch_size);
//     toutput = tensor_alloc(batch_size, seq_len,n_out);
// }

// void SqLayerDense::forward(Mat inputs)
// {
//     layerinputs = inputs;
//     if (layerbias)
//     {
//         power_mat_dot_bias(output, inputs, weights, biases, pow);
//     }
//     else
//     {
//         power_mat_dot(output, inputs, weights, pow);
//     }
// }

// void SqLayerDense::forward(Tensor inputs)
// {
    
// }

// void SqLayerDense::backward(Mat dvalues)
// {
//     mat_transpose(layerinputs_T,  layerinputs);
//     mat_dot(dweights, layerinputs_T, dvalues);
    
//     std::cout << std::endl;
//     mat_scalertimes(doubleweights, weights, 2);
//     mat_elemwise_mult(dweights, dweights, doubleweights);
   
    
//     // need to figure out what this part means......
//     if (layerbias)
//     {
//        for (size_t i = 0; i < dbiases.rows; i++)
//        {
//         MAT_AT(dbiases, i, 0) = 0;
//            for (size_t j = 0; j < dvalues.cols; j++)
//            {
//                 MAT_AT(dbiases, i, 0) += MAT_AT(dvalues, i, j);
//            }   
//        }
//     }
//     //...........
//     mat_transpose(weights_T, weights);
//     mat_elemwise_mult(weightsTsq, weights_T, weights_T);
//     mat_dot(dinputs, dvalues, weightsTsq);
// }



// class CubeLayerDense
// {
//     CubeLayerDense(int n_inputs, int n_out, bool bias, int num_samples);
//     void forward(Mat inputs);
//     // void forward(Tensor inputs);
//     void backward(Mat dvalues);
//     // void backward(Tensor dvalues);
//     void randomize(float percentrand);
//     //~LayerDense();
//     Mat output;
//     Tensor toutput;
//     Mat dinputs;
//     Mat dbiases;

// //private:
//     Mat weights;
//     Mat weights_T;
//     Mat biases;
//     bool layerbias;
//     int n_samples;
    
//     Mat dweights;
    
//     Mat layerinputs;
//     Mat layerinputs_T;
//     Tensor tlayerinputs;

//     Mat weightscubederiv;
//     Mat weightsTcube;
//     Mat weightsSq;
//     Mat tripleweightsSq;
//     float pow;
// };


// CubeLayerDense::CubeLayerDense(int n_inputs, int n_out, bool bias, int num_samples)
// {
//     n_samples = num_samples;
//     pow = 3;
//     layerbias = bias;
//     weights = mat_alloc(n_inputs, n_out);
//     mat_rand(weights, -0.1, 0.1);
//     weights_T = mat_alloc(n_out, n_inputs);
//     if (layerbias)
//     {
//         biases = mat_alloc(1,n_out);
//         mat_rand(biases, -1.0,1.0);
//         dbiases = mat_alloc(1, n_out);
//     }
//     dweights = mat_alloc(n_inputs, n_out);
//     tripleweightsSq = mat_alloc(n_inputs, n_out);
//     dinputs = mat_alloc(num_samples,n_inputs);
//     layerinputs = mat_alloc(num_samples, n_inputs);
//     layerinputs_T = mat_alloc(n_inputs, num_samples);
//     output = mat_alloc(n_samples,n_out);

//     weightsTcube = mat_alloc(n_out, n_inputs);
//     weightsSq = mat_alloc(n_inputs, n_out);
// }


// void CubeLayerDense::forward(Mat inputs)
// {
//      layerinputs = inputs;
//     if (layerbias)
//     {
//         power_mat_dot_bias(output, inputs, weights, biases, pow);
//     }
//     else
//     {
//         power_mat_dot(output, inputs, weights, pow);
//     }
// }

// //not done
// void CubeLayerDense::backward(Mat dvalues)
// {
//     mat_transpose(layerinputs_T,  layerinputs);
//     mat_dot(dweights, layerinputs_T, dvalues);

//     mat_scalertimes(tripleweightsSq, weights, 3);
//     mat_elemwise_mult(tripleweightsSq, weights, tripleweightsSq);
//     mat_elemwise_mult(dweights, dweights, tripleweightsSq);
//     // need to figure out what this part means......
//     if (layerbias)
//     {
//        for (size_t i = 0; i < dbiases.rows; i++)
//        {
//         MAT_AT(dbiases, i, 0) = 0;
//            for (size_t j = 0; j < dvalues.cols; j++)
//            {
//                 MAT_AT(dbiases, i, 0) += MAT_AT(dvalues, i, j);
//            }   
//        }
//     }
   
//     mat_transpose(weights_T, weights);
//     //double elem wise mult to get a cubed value. w_t*w_t*w_t.
//     mat_elemwise_mult(weightsTcube, weights_T, weights_T);
//     mat_elemwise_mult(weightsTcube, weightsTcube, weights_T);
//     mat_dot(dinputs, dvalues, weightsTcube);
// }




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
    // Relu_Activation(int num_samples, int num_neurons, int numdvalrows, int numdvalcols);
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

// Relu_Activation::Relu_Activation(int num_samples, int num_neurons, int numdvalrows, int numdvalcols)
// {
//     dinputs = mat_alloc(num_samples, num_neurons);
//     output = mat_alloc(num_samples, num_neurons);
//     layerinputs = mat_alloc(num_samples, num_neurons);
//     dinputs = mat_alloc(numdvalrows, numdvalcols);
// }

void Relu_Activation::forward(Mat inputs)
{
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
    Sigmoid(int num_samples, int n_out);
    void forward(Mat inputs);
    void backward(Mat dvalues);
    //~Sigmoid();
    Mat output;
    Mat dinputs;
};

Sigmoid::Sigmoid(int num_samples, int n_out)
{
    output = mat_alloc(num_samples, n_out);
    dinputs = mat_alloc(num_samples, n_out);
}

void Sigmoid::forward(Mat inputs)
{
    // output = mat_alloc(inputs.rows, inputs.cols);
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
    // dinputs = mat_alloc(dvalues.rows, dvalues.cols);
    
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
    Activation_softmax(int num_samples, int num_classes);
    Activation_softmax(int batch_size, int seq_len, int num_classes);
    void forward(Mat inputs);
    void backward(Mat dvalues);

    std::vector<Mat> jacobians;
    Mat diagflat;
    Mat output;
    Mat dinputs;
    Mat temp;
    Mat temp_t;
    Mat singleoutput;

    //here......
    Mat singledval;
    //end...........


    Mat singleoutput_T;
    Mat soutdot;

    Tensor toutput;
    Tensor tdinputs;

    Mat exp_vals;
    Mat exp_sum;
    Mat maxes;

    Mat dval_T;
    Mat tempjac;
};



Activation_softmax::Activation_softmax(int num_samples, int num_classes)
{
    dinputs = mat_alloc(num_samples,num_classes);
    // diagflat = mat_alloc(num_classes,num_classes);

    // mat_fill(diagflat, 0.0);
    // temp = mat_alloc(num_classes,1);
    // temp_t = mat_alloc(1,num_classes);

    exp_vals = mat_alloc(num_samples, num_classes);
    exp_sum = mat_alloc(num_samples, 1);
    maxes = mat_alloc(num_samples, 1);
    output = mat_alloc(num_samples,  num_classes);
}

Activation_softmax::Activation_softmax(int batch_size, int seq_len, int num_classes)
{
    tdinputs = tensor_alloc(batch_size, seq_len,num_classes);
    diagflat = mat_alloc(num_classes,num_classes);

    mat_fill(diagflat, 0.0);
    temp = mat_alloc(num_classes,1);
    temp_t = mat_alloc(1,num_classes);

    exp_vals = mat_alloc(batch_size, num_classes);
    exp_sum = mat_alloc(batch_size, 1);
    maxes = mat_alloc(batch_size, 1);
    toutput = tensor_alloc(batch_size,seq_len,  num_classes);
    singleoutput = mat_alloc(1, num_classes);
    
    singleoutput_T = mat_alloc(num_classes, 1);
    soutdot = mat_alloc(num_classes, num_classes);
   dval_T = mat_alloc(num_classes,1);
//    tempjac = mat_alloc(num_classes, num_classes);
   for (size_t i = 0; i < batch_size; i++)
   {
        jacobians.push_back(mat_alloc(num_classes, num_classes));
   }
}

void Activation_softmax::forward(Mat inputs)
{   
    
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
    
    mat_fill(exp_sum, 0.0);
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
    for (size_t i = 0; i < output.rows; i++)
    {
        float dot(0);
        
        for (size_t j = 0; j < output.cols; j++)
        {
            dot += MAT_AT(output, i,j) * MAT_AT(dvalues, i, j);
        }

        for (size_t j = 0; j < output.cols; j++)
        {
            MAT_AT(dinputs, i, j) = MAT_AT(output, i, j) * (MAT_AT(dvalues, i, j) - dot);
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
    // int labels = dvalues.cols;
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
    BinaryCrossentropy_Loss(int num_samples, int num_cols);
    void forward(Mat y_pred, Mat y_true);
    void backward(Mat dvalues, Mat y_true);

    Mat sample_losses;
    Mat y_pred_clipped;
    Mat clipped_dvalues;
    Mat dinputs;
};

BinaryCrossentropy_Loss::BinaryCrossentropy_Loss(int num_samples, int num_cols)
{
    y_pred_clipped = mat_alloc(num_samples, num_cols);
    sample_losses = mat_alloc(num_samples, num_cols);

    //for backward
    clipped_dvalues = mat_alloc(num_samples, num_cols);
    dinputs = mat_alloc(num_samples, num_cols);
}


void BinaryCrossentropy_Loss::forward(Mat y_pred, Mat y_true)
{
    
    for (size_t i = 0; i < y_pred.rows; i++)
    {
        
        MAT_AT(y_pred_clipped,i,0) = std::clamp(MAT_AT(y_pred,i,0),0.0001, 0.999);
        MAT_AT(sample_losses, i, 0) = -1*((MAT_AT(y_true, i, 0)*std::log(MAT_AT(y_pred_clipped, i, 0))) + ((1 - MAT_AT(y_true, i, 0))*std::log(1 - MAT_AT(y_pred_clipped,i,0))));
    }
}

void BinaryCrossentropy_Loss::backward(Mat dvalues, Mat y_true)
{
   
    for (size_t i = 0; i < dvalues.rows; i++)
    {
        MAT_AT(clipped_dvalues, i,0) = std::clamp(MAT_AT(dvalues,i,0),0.0001, 0.999);
        // MAT_AT(dinputs,i,0) = -(MAT_AT(y_true,i,0)/MAT_AT(clipped_dvalues,i,0) - (1-MAT_AT(y_true,i,0))/(1-MAT_AT(clipped_dvalues,i,0)));
        MAT_AT(dinputs,i,0) = (MAT_AT(clipped_dvalues, i, 0) - MAT_AT(y_true, i, 0)) /(MAT_AT(clipped_dvalues, i, 0) * (1 - MAT_AT(clipped_dvalues, i, 0)));
        MAT_AT(dinputs,i,0) /= dvalues.rows;

    }
    
}


class AttentionHead
{
public:
    AttentionHead(int num_embed, int batch_size, int seq_len, int head_dim);
    void forward(const Tensor& x);
    void backward(const Tensor& dvalues);

    Tensor output;
    Mat w_q;
    Mat w_k;
    Mat w_v;

    Tensor d_x;
    Mat d_w_q;
    Mat d_w_k;
    Mat d_w_v;
// private:
    int num_embed;
    int head_dim;

    Mat v_T;
    Mat att_weight_T;
    Mat d_attention_weights;
    Mat d_v;

    std::map<std::string, Tensor> cache;
    Activation_softmax soft;
    
};

AttentionHead::AttentionHead(int num_embed, int batch_size, int seq_len, int head_dim) 
    : num_embed(num_embed), head_dim(head_dim), soft(seq_len, seq_len)
{
    w_q = mat_alloc(num_embed, head_dim);
    w_k = mat_alloc(num_embed, head_dim);
    w_v = mat_alloc(num_embed, head_dim);

    mat_rand(w_q, -0.5, 0.5);
    mat_rand(w_k, -0.5,0.5);
    mat_rand(w_v, -0.5,0.5);

    //size batch_size, seq_len, head_dim
    cache["q"] = tensor_alloc(batch_size,  seq_len, head_dim);
    cache["k"] = tensor_alloc(batch_size,  seq_len, head_dim);
    cache["v"] = tensor_alloc(batch_size,  seq_len, head_dim);
    //size batch_size, seq_len, seq_len
    cache["attention_weights"] = tensor_alloc(batch_size,  seq_len, seq_len);
    Activation_softmax soft(seq_len, seq_len);
    output = tensor_alloc(batch_size, seq_len, head_dim);

    //for backward pass
    d_x = tensor_alloc(batch_size, seq_len, num_embed);
    d_w_k = mat_alloc(num_embed, head_dim);
    d_w_q = mat_alloc(num_embed, head_dim);
    d_w_v = mat_alloc(num_embed, head_dim);

    //in the future should probably replace this with just calculating and forwarding to other gradients in the backward pass rather
    //than saving these weights. 
    d_attention_weights = mat_alloc(seq_len, seq_len);
    v_T = mat_alloc(head_dim, seq_len);
    att_weight_T = mat_alloc(seq_len, seq_len);
    d_v = mat_alloc(seq_len, head_dim);

   

}


void AttentionHead::forward(const Tensor& x)
{
    size_t batch_size = x.depth;
    int seq_len = x.rows;

    cache["x"] = x;

    for (size_t b = 0; b < batch_size; b++)
    {
        mat_dot(cache["q"].mats[b], x.mats[b], w_q);
        mat_dot(cache["k"].mats[b], x.mats[b], w_k);
        mat_dot(cache["v"].mats[b], x.mats[b], w_v);


        Mat scores = mat_alloc(seq_len, seq_len);
        Mat kt = mat_alloc(cache["k"].cols,cache["k"].rows);
        mat_dot(scores, cache["q"].mats[b], kt);
        mat_scalertimes(scores, scores, (1/std::sqrt(static_cast<double>(head_dim))));

        //next apply softmax to scores to get attention weights.
        soft.forward(scores);
        cache["attention_weights"].mats[b] = soft.output;

        mat_dot(output.mats[b], cache["attention_weights"].mats[b], cache["v"].mats[b]);

        mat_free(scores);
        mat_free(kt);
    }
}

// in future update, make tensor a class and change multipl. transpose and handle memory management in destructor, also copy assignment etc. 
//right now can only work for  vision because it has fixed sequence length.
void AttentionHead::backward(const Tensor& dvalues)
{
    const Tensor& x  = cache["x"];
    const Tensor& q = cache["q"];
    const Tensor& k = cache["k"];
    const Tensor& v = cache["v"];
    const Tensor& att_weights = cache["attention_weights"];

    size_t batch_size = x.depth;
    mat_fill(d_w_q, 0);
    mat_fill(d_w_k, 0);
    mat_fill(d_w_v, 0);

    for (size_t b = 0; b < batch_size; b++)
    {
        int awrows = att_weights.mats[b].rows;
        int awcols = att_weights.mats[b].cols;
        Mat d_q = mat_alloc(q.rows, q.cols);
        Mat d_k = mat_alloc(k.rows, k.cols);
        
        //4. gradient of weighted sum of values
        mat_transpose(v_T, v.mats[b]);
        mat_dot(d_attention_weights, dvalues.mats[b],v_T);
        mat_transpose(att_weight_T, att_weights.mats[b]);
        mat_dot(d_v, att_weight_T, dvalues.mats[b]);
        soft.backward(att_weights.mats[b]);
        
        Mat dsc_T = mat_alloc(soft.dinputs.cols, soft.dinputs.rows);
        mat_transpose(dsc_T, soft.dinputs);
        //2. gradient of attention scores
        for (size_t j = 0; j < soft.dinputs.rows; j++)
        {
            for (size_t k = 0; k < soft.dinputs.cols; k++)
            {
                MAT_AT(soft.dinputs, j, k) = MAT_AT(soft.dinputs, j,k) / std::sqrt(static_cast<double>(head_dim));
                
            }
        }
        mat_dot(d_q, soft.dinputs, k.mats[b]);
        mat_dot(d_k, dsc_T, q.mats[b]);


        //1. gradient of input projections
        Mat xb_T = mat_alloc(x.mats[b].cols, x.mats[b].rows);
        Mat tempd = mat_alloc(xb_T.rows, d_q.cols); //just chose d_w_q because its first, will fit any d_wq, d_wk,d_wv
        mat_dot(tempd, xb_T, d_q);
        mat_sum(d_w_q, tempd);

        mat_fill(tempd, 0);
        mat_dot(tempd, xb_T, d_k);
        mat_sum(d_w_k, tempd);

        mat_fill(tempd, 0);
        mat_dot(tempd, xb_T, d_v);
        mat_sum(d_w_v, tempd);


        //gradient wrt input x
        Mat weights_t = mat_alloc(w_q.cols, w_q.rows);   //once again just chose one of the three options wq, wk, wv
        Mat tempw = mat_alloc(d_q.rows, weights_t.cols);
        mat_transpose(weights_t, w_q);
        
        mat_dot(tempw, d_q, weights_t);
        mat_sum(d_x.mats[b], tempw);
        mat_fill(tempw, 0);
        
        mat_transpose(weights_t, w_k);
        mat_dot(tempw, d_k, weights_t);
        mat_sum(d_x.mats[b], tempw);
        mat_fill(tempw, 0);

        mat_transpose(weights_t, w_v);
        mat_dot(tempw, d_v, weights_t);
        mat_sum(d_x.mats[b], tempw);


        //free all temporary mats
        mat_free(d_k);
        mat_free(d_q);
        // mat_free(dscores);
        mat_free(dsc_T);
        mat_free(xb_T);
        mat_free(tempd);
        mat_free(weights_t);
        mat_free(tempw);
    }
    

}



class MultiheadAttention
{
public:
    MultiheadAttention(int num_heads, int num_embed, int batch_size, int seq_len);
    void forward(const Tensor& x);
    void copyheads();
    void backward(const Tensor& dvalues);

    Tensor output;
    Tensor headsoutput;
    int head_dim;
    std::vector<AttentionHead> heads;
    int bsize, slen, emb_dim, n_heads;
    LayerDense den1;
};


MultiheadAttention::MultiheadAttention(int num_heads, int num_embed, int batch_size, int seq_len) : den1(num_embed, num_embed, false, batch_size, seq_len)
{
    bsize = batch_size;
    slen = seq_len;
    emb_dim = num_embed;
    n_heads = num_heads;
    assert(num_embed % num_heads == 0);
    head_dim = num_embed / num_heads;

    for (size_t i = 0; i < num_heads; i++)
    {
        heads.push_back(AttentionHead(num_embed, batch_size, seq_len, head_dim));
    }
    headsoutput = tensor_alloc(batch_size, seq_len, num_embed);
    output = tensor_alloc(batch_size, seq_len, num_embed);
    //i guess create dropout layer here...? yes i should learn this anyway. 
    

}

void MultiheadAttention::copyheads()
{
    for (size_t i = 0; i < bsize; i++)
    {
        for (size_t j = 0; j < slen; j++)
        {
            int nh = 0;
            for(auto head : heads)
            {
                for (size_t k = 0; k < head_dim; k++)
                {
                    MAT_AT(headsoutput.mats[i], j, (nh*head_dim)+k) = MAT_AT(head.output.mats[i], j, k);
                }
                nh++;
            }
        }
        
    }
    
}

void MultiheadAttention::forward(const Tensor& x)
{
    for(auto head : heads)
    {
        head.forward(x);
    }
    //copy output of each head into a concatenated output.

    copyheads();
    den1.forward(headsoutput);
    output = headsoutput;
}

class Optimizer_SGD
{
public:
    Optimizer_SGD(double learning_rate=0.01){  lr = learning_rate;}
    void update_params(LayerDense Layer);
    void update_params(Lora Layer);
    // void update_params(SqLayerDense layer);
    void update_params(Convolution2D conv);
    void update_params(AttentionHead layer);
    void update_params(LayerKron layer);
    void update_randomly(LayerDense layer, float L);

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

void Optimizer_SGD::update_params(Lora layer)
{
    for (size_t i = 0; i < layer.wA.rows; i++)
    {
        for (size_t j = 0; j < layer.wA.cols; j++)
        {
            // std::cout << "before " << MAT_AT(layer.wA, i, j) << "\n";
            MAT_AT(layer.wA, i, j) -=  lr * MAT_AT(layer.dwA, i, j);
            // std::cout <<
        }
    }
    for (size_t i = 0; i < layer.wB.rows; i++)
    {
        for (size_t j = 0; j < layer.wB.cols; j++)
        {
            MAT_AT(layer.wB, i, j) -=  lr * MAT_AT(layer.dwB, i, j);
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


// void Optimizer_SGD::update_params(SqLayerDense layer)
// {
//     for (int i = 0; i < layer.weights.rows; i++)
//     {
//         for (int j = 0; j < layer.weights.cols; j++)
//         {
//             MAT_AT(layer.weights, i, j) -= lr* MAT_AT(layer.dweights, i, j);
//         }
//     }
//
//     if (layer.layerbias)
//     {
//         for (int i = 0; i < layer.biases.cols; i++)
//         {
//             MAT_AT(layer.biases, 0, i) -= lr * MAT_AT(layer.dbiases, 0, i);
//         }
//     }
// }




void Optimizer_SGD::update_params(Convolution2D conv)
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


void Optimizer_SGD::update_params(AttentionHead layer)
{
    for (size_t i = 0; i < layer.w_k.rows; i++)
    {
        for (size_t j = 0; j < layer.w_k.cols; j++)
        {
            MAT_AT(layer.w_k, i,j) -= lr * MAT_AT(layer.d_w_k, i, j); 
        }
    }
    for (size_t i = 0; i < layer.w_q.rows; i++)
    {
        for (size_t j = 0; j < layer.w_q.cols; j++)
        {
            MAT_AT(layer.w_q, i,j) -= lr * MAT_AT(layer.d_w_q, i, j); 
        }
    }
    for (size_t i = 0; i < layer.w_v.rows; i++)
    {
        for (size_t j = 0; j < layer.w_v.cols; j++)
        {
            MAT_AT(layer.w_v, i,j) -= lr * MAT_AT(layer.d_w_v, i, j); 
        }
    }
    
}


void Optimizer_SGD::update_params(LayerKron layer)
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
        for (size_t j = 0; j < layer.biases.rows; j++)
        {
            for (int i = 0; i < layer.biases.cols; i++)
            {
                MAT_AT(layer.biases, j, i) -= lr * MAT_AT(layer.dbiases, j, i);
            }
        }
    }
}



void Optimizer_SGD::update_randomly(LayerDense layer, float L)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // Define the distribution for the float numbers (e.g., between 0.0 and 1.0)
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    // Generate and print a random float
    
    for (int i = 0; i < layer.weights.rows; i++)
    {
        for (int j = 0; j < layer.weights.cols; j++)
        {
            float randomFloat = distribution(generator);
            MAT_AT(layer.weights, i, j) += randomFloat * L;        //rand * L
        }
    }

    if (layer.layerbias)
    {
        for (int i = 0; i < layer.biases.cols; i++)
        {
            float randomFloat = distribution(generator);
            MAT_AT(layer.biases, 0, i)  += randomFloat * L;  //rand * L
        }
    }
}










//i dont think this is used anywhere............

// class AttentionDense
// {
// public:
//     AttentionDense(int n_inputs, int n_out, bool bias);

//     void forward(Tensor inputs);
//     void halfforward(Mat inputs);
//     void halfbackward(Mat dvalues);
//     void backward(Mat dvalues);
//     void backward(Tensor dvalues);
   
//     Mat output;
    
//     Tensor toutput;
//     Mat dinputs;
//     Mat dbiases;

// //private:
//     Mat weights;
//     Mat weights_T;
//     Mat biases;
//     bool layerbias;
    
//     Mat dweights;
//     Tensor layerinputs;
    
//     Tensor tlayerinputs;
// };


// AttentionDense::AttentionDense(int n_inputs, int n_out, bool bias)
// {
//     layerbias = bias;
//     weights = mat_alloc(n_inputs, n_out);
//     mat_rand(weights, -0.1, 0.1);
//     weights_T = mat_alloc(n_out, n_inputs);
//     if (layerbias)
//     {
//         biases = mat_alloc(1,n_out);
//         mat_rand(biases, -1.0,1.0);
//         dbiases = mat_alloc(1, n_out);
//     }
//     dweights = mat_alloc(n_inputs, n_out);
//     // dinputs = mat_alloc(num_samples,n_inputs);
//     // layerinputs = mat_alloc(num_samples, n_inputs);
//     // layerinputs_T = mat_alloc(n_inputs, num_samples);
//     // output = mat_alloc(n_samples,n_out);
//     // duboutput = mat_alloc(n_samples, 2*n_out);
//     Tensor dinputs;
// }

// void AttentionDense::forward(Tensor inputs)
// {
//     tlayerinputs = inputs;
//     if (layerbias)
//     {
//         //bmm(toutput, inputs, weights);
//         std::cout <<  "did not make tensor version of dot product with bias in layerdense \n";
//     }
//     else
//     {
//         bmm(toutput, inputs, weights);
        
//     }
// }

// void AttentionDense::backward(Mat dvalues)
// {
//     //must rework this..............
//     // mat_dot(dweights, layerinputs, dvalues);
//     //................
//     if (layerbias)
//     {
//         for (size_t i = 0; i < dvalues.cols; i++)
//         {
//             for (size_t j = 0; j < dvalues.rows; j++)
//             {
//                 MAT_AT(dbiases, 0, j) += MAT_AT(dvalues, j, i);
//             }
            
//         }
//     }
//     //...........
//     mat_transpose(weights_T, weights);
    
//     mat_dot(dinputs, dvalues, weights_T);
// }

