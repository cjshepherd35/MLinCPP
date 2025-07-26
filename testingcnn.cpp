#include <iostream>
#include "nnfromscratchfrompyversion.cpp"
#include <fstream>
#include <time.h>
#include <random>



std::string labelsfilename = "C:/Users/cshep/Downloads/t10k-labels.idx1-ubyte";
std::string imagesfilename = "C:/Users/cshep/Downloads/t10k-images.idx3-ubyte/t10k-images.idx3-ubyte";

int num_samples = 6;
double lr = 0.01;
// int n_inputs = 28*28;
int  n_neurons = 256;
int num_classes = 10;
int num_iters = 201;
int check_iter = 50;
int kernel_size = 5;
int input_height = 28;
int input_width = 28;
int depth = 6;


//reading mnist images
std::vector<std::vector<float>> read_images(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);

    char magicNumber[4];
    char numImages[4];
    char numRows[4];
    char numCols[4];
    file.read(magicNumber, 4);
    file.read(numImages, 4);
    file.read(numRows, 4);
    file.read(numCols, 4);

    int numims = (static_cast<unsigned char> (numImages[0]) << 24) | (static_cast<unsigned char> (numImages[1]) << 16) | (static_cast<unsigned char> (numImages[2]) << 8) | static_cast<unsigned char> (numImages[3]);
    int numrs =  (static_cast<unsigned char> (numRows[0]) << 24) | (static_cast<unsigned char> (numRows[1]) << 16) | (static_cast<unsigned char> (numRows[2]) << 8) | static_cast<unsigned char> (numRows[3]);
    int numcs = (static_cast<unsigned char> (numCols[0]) << 24) | (static_cast<unsigned char> (numCols[1]) << 16) | (static_cast<unsigned char> (numCols[2]) << 8) | static_cast<unsigned char> (numCols[3]);

    std::vector<std::vector<unsigned char>> cimages;

    for (size_t i = 0; i < numims; i++)
    {
        std::vector<unsigned char> image(numrs*numcs);
        file.read((char*)(image.data()), numrs*numcs);
        cimages.push_back(image);
    }
    file.close();

    std::vector<std::vector<float>> images;
    
    for (size_t i = 0; i < cimages.size(); i++)
    {
        std::vector<float> tempim;
        for (size_t j = 0; j < cimages[0].size(); j++)
        {
            tempim.push_back((float)cimages[i][j]/255.f);
        }
        images.push_back(tempim);
    }
    return images;
}

//read in labels for the images.
std::vector<std::vector<int>> read_labels(const std::string& filename)
{
    std::vector<std::vector<unsigned char>> clabels;
    std::ifstream file(filename, std::ios::binary);

    char magicNumber[4];
    char numLabels[4];
    file.read(magicNumber, 4);
    file.read(numLabels,4);
    int numlabs = (static_cast<unsigned char> (numLabels[0]) << 24) | (static_cast<unsigned char> (numLabels[1]) << 16) | (static_cast<unsigned char> (numLabels[2]) << 8) | static_cast<unsigned char> (numLabels[3]);
    for (size_t i = 0; i < numlabs; i++)
    {
        std::vector<unsigned char> label(1);
        file.read((char*)(label.data()),sizeof(char));
        clabels.push_back(label);
    }
    file.close();

    std::vector<std::vector<int>> labels;
    for (size_t i = 0; i < numlabs; i++)
    {
        std::vector<int> templabel;
        for (size_t j = 0; j < num_classes; j++)
        {
            if (j == (int)clabels[i][0])
            {
                templabel.push_back(1);
            }  
            else templabel.push_back(0);
        }
        labels.push_back(templabel);
    }

    return labels;
}

std::vector<int> getrandvec(int size)
{
    std::vector<int> randvec;
    
    for (size_t i = 0; i < num_samples; i++)
    {
        randvec.push_back(rand() % 10000);
    }
    return randvec;
}

std::vector<Mat> randomizeImages(std::vector<std::vector<float>> images, std::vector<int> randvec)
{
    
    std::vector<Mat> trainimages;
    for (size_t i = 0; i < num_samples; i++)
    {
        trainimages.push_back(mat_alloc(input_height, input_width));
    }
    
    //change this...........
    int i = 0;
    for (int randnum : randvec)
    {
        for (size_t j = 0; j < input_height; j++)
        {
            for (size_t k = 0; k < input_width; k++)
            {
                MAT_AT(trainimages[i], j,k) = images[randnum][j*input_height+k];
            }
            
        }
        
        i++;
    }
    return trainimages;
}

Mat randomizeLabels(std::vector<std::vector<int>> labels, std::vector<int> randvec)
{
    Mat trainlabels = mat_alloc(num_samples, num_classes);
    int i = 0;
    for (int randnum : randvec)
    {
        for (size_t k = 0; k < num_classes; k++)
        {
            MAT_AT(trainlabels, i,k) = labels[randnum][k]; 
        }
        i++;
    }
    return trainlabels;
}

void ReshapeConv(std::vector<std::vector<Mat>> convoutput, Mat reshapeoutput)
{   
    int rows = convoutput[0][0].rows;
    int cols = convoutput[0][0].cols;
    // Mat reshape = mat_alloc(convoutput.size(), convoutput[0].size()*convoutput[0][0].rows*convoutput[0][0].cols);
    for (size_t i = 0; i < convoutput.size(); i++)
    {
        for (size_t j = 0; j < convoutput[0].size(); j++)
        {
            for (size_t k = 0; k < convoutput[0][0].rows; k++)
            {
                for (size_t l = 0; l < convoutput[0][0].cols; l++)
                {
                    MAT_AT(reshapeoutput, i,j*rows*cols + cols*k + l) = MAT_AT(convoutput[i][j], k,l);
                }
            }
        }
    }
}

void reshapeToConv(Mat dinput, std::vector<std::vector<Mat>> reshapedinput)
{
    int rows = reshapedinput[0][0].rows;
    int cols = reshapedinput[0][0].cols;
    for (size_t i = 0; i < reshapedinput.size(); i++)
    {
        for (size_t j = 0; j < reshapedinput[0].size(); j++)
        {
            for (size_t k = 0; k < reshapedinput[0][0].rows; k++)
            {
                for (size_t l = 0; l < reshapedinput[0][0].cols; l++)
                {
                     MAT_AT(reshapedinput[i][j], k,l) = MAT_AT(dinput, i,j*rows*cols + cols*k + l);
                }
            }
        }
    }
}

int main() {
    srand(69);
    std::vector<std::vector<float>> images = read_images(imagesfilename);
    std::vector<std::vector<int>> labels = read_labels(labelsfilename);
    
    //change this..........................
    // Mat trainimages = mat_alloc(num_samples, n_inputs);
    std::vector<Mat> trainimages;
    
    Mat trainlabels = mat_alloc(num_samples, num_classes);
    Mat flatconv = mat_alloc(num_samples, depth*(input_height-kernel_size+1)*(input_width-kernel_size+1));
    std::vector<std::vector<Mat>> backToConvShape;
    for (size_t i = 0; i < num_samples; i++)
    {
        std::vector<Mat> tempmat;
        for (size_t j = 0; j < depth; j++)
        {
            tempmat.push_back(mat_alloc(input_height-kernel_size+1, input_width-kernel_size+1));
        }
        backToConvShape.push_back(std::move(tempmat));
    }
    

    Convolution2D conv(depth, kernel_size, num_samples, input_height, input_width);
    Relu_Activation relu(num_samples, depth*input_height*input_width);
    LayerDense ld2(depth*input_height*input_width, n_neurons, true, num_samples);
    Relu_Activation relu2(num_samples, n_neurons, num_samples, n_neurons);
    LayerDense ld3(n_neurons, num_classes, true, num_samples);
    
    Activation_softmax soft(num_samples, num_classes, n_neurons);
    Loss_categoricalCrossentropy loss(num_classes, num_samples);
    

    Optimizer_SGD opt(lr);
    std::vector<int> randvec;
    for (size_t i = 0; i < num_iters; i++)
    {
        randvec = getrandvec(num_samples);
        trainimages = randomizeImages(images, randvec);
        trainlabels = randomizeLabels(labels, randvec);
        conv.forward(trainimages);
        //need reshape here numsamplex(conv_depth*height*width)
        ReshapeConv(conv.output, flatconv);
        relu.forward(flatconv);
        ld2.forward(relu.output);
        relu2.forward(ld2.output);
        ld3.forward(relu2.output);
        soft.forward(ld3.output);
        loss.forward(soft.output, trainlabels);
        

        loss.backward(soft.output, trainlabels);
        if (i% check_iter ==0 )
        {
            printf("iter: %d\n", i);
            float total_loss = 0.0;
            
            for (size_t l = 0; l < loss.negative_log_likelihood.cols; l++)
            {
                total_loss += MAT_AT(loss.negative_log_likelihood, 0, l);
            }
            float avg_loss = total_loss/num_samples;
            printf("average loss: %f\n", avg_loss);
        }

        soft.backward(loss.dinputs);
        
        ld3.backward(soft.dinputs);
        relu2.backward(ld3.dinputs);
        ld2.backward(relu2.dinputs);
        relu.backward(ld2.dinputs);
        reshapeToConv(relu.dinputs, backToConvShape);
        conv.backward(backToConvShape);

        opt.update_params(ld3);
        opt.update_params(ld2);
        opt.update_convparams(conv);
    }
    
    
    
    return 0;
}