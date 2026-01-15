#include <iostream>
#include "../nnfromscratchfrompyversion.cpp"
#include <fstream>
#include <time.h>
#include <random>


double lr = 0.01;
int height = 28;
int width = 28;
int  n_neurons = 256;
int num_classes = 10;
int num_iters = 201;
int check_iter = 50;
int numpatches = 16;
int patchsize = 7;
int batch_size = 32;
int seq_len = numpatches; //get rid of because this will be numpatches.
int embed_dim = patchsize*patchsize;
int num_heads = 1;


Tensor images;
Tensor trainimages = tensor_alloc(batch_size, height, width);
Tensor patchImages = tensor_alloc(batch_size, numpatches, embed_dim);


std::string labelsfilename = "C:/Users/cshep/Downloads/t10k-labels.idx1-ubyte";
std::string imagesfilename = "C:/Users/cshep/Downloads/t10k-images.idx3-ubyte/t10k-images.idx3-ubyte";

//reading mnist images
void read_images(const std::string& fileName)
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

    images = tensor_alloc(numims, numrs, numcs);
    for (size_t i = 0; i < cimages.size(); i++)
    {
        
        for (size_t j = 0; j < cimages[0].size(); j++)
        {
           
          MAT_AT(images.mats[i], j/height, j%height) = (float)cimages[i][j]/255.f;
           
        }
    }
    
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
    
    for (size_t i = 0; i < batch_size; i++)
    {
        randvec.push_back(rand() % 10000);
    }
    return randvec;
}

void randomizeandPatchifyImages(std::vector<Mat> images, std::vector<int> randvec)
{
    
    
    int i = 0;
    for (int randnum : randvec)
    {
        trainimages.mats[i] = images[randnum];
        i++;
    }

    for (size_t b = 0; b < batch_size; b++)
    {
        
        for (size_t j = 0; j < numpatches; j++)
        {
            for (size_t k = 0; k < patchsize; k++)
            {
                for (size_t l = 0; l < patchsize; l++)
                {
                    MAT_AT(patchImages.mats[b], j, k*patchsize+l) = MAT_AT(trainimages.mats[b],(j/4)*patchsize+k,7*(j%4)+l);
                }   
            }
        }
    }
    
    
    
}


Mat randomizeLabels(std::vector<std::vector<int>> labels, std::vector<int> randvec)
{
    Mat trainlabels = mat_alloc(batch_size, num_classes);
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




int main()
{
    srand(37);

    read_images(imagesfilename);
    std::vector<std::vector<int>> labels = read_labels(labelsfilename);
    std::vector<int> randvec;

    
    // AttentionHead sa(embed_dim, batch_size, seq_len, embed_dim);
    MultiheadAttention mha(num_heads, embed_dim, batch_size, seq_len);
    LayerDense den1(embed_dim, num_classes, false, batch_size, seq_len);
    Activation_softmax soft(batch_size, num_classes);
    Optimizer_SGD opt(lr = lr);
    // Mat den1outputreshape = mat_alloc(batch_size*seq_len, embed_dim);
    randvec = getrandvec(batch_size);
    randomizeandPatchifyImages(images.mats, randvec);
    // sa.forward(patchImages);
    // std::cout << "out " << sa.output.depth << " x " << sa.output.rows << " x " << sa.output.cols << std::endl; 
    mha.forward(patchImages);
    den1.forward(mha.output);


    //now need to add to softmax function to accomadate tensors............
    

    // for (size_t i = 0; i < 100; i++)
    // {
    //     randvec = getrandvec(batch_size);
    //     randomizeandPatchifyImages(images.mats, randvec);
    //     sa.forward(patchImages);
    //     sa.backward(sa.output);
    // }
    // opt.update_params(sa);
    
    return 0;
}