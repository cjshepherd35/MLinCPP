//only update a few weights in each layer and see if we still get the same learning. must make all gradients to pass back, will see later if we can skip some grads. 
#include <iostream>
#include <fstream>
#include <time.h>
#include <random>
#include <chrono>
#include "../nnfromscratchfrompyversion.cpp"


std::string labelsfilename = "C:/Users/cshep/Downloads/t10k-labels.idx1-ubyte";
std::string imagesfilename = "C:/Users/cshep/Downloads/t10k-images.idx3-ubyte/t10k-images.idx3-ubyte";


int num_samples = 10;
double lr = 0.1;
int n_inputs = 28*28;
int  n_neurons = 128;
int num_classes = 10;
int num_iters = 501;
int check_iter = 50;
double percchangeableweights = 0.5;

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

Mat randomizeImages(std::vector<std::vector<float>> images, std::vector<int> randvec)
{
    Mat trainimages = mat_alloc(num_samples, n_inputs);
    int i = 0;
    for (int randnum : randvec)
    {
        for (size_t j = 0; j < n_inputs; j++)
        {
            MAT_AT(trainimages, i,j) = images[randnum][j];  
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

class Fewopt
{
public:
    Fewopt(std::vector<std::vector<std::tuple<int, int>>> randgrads, double learningrate=0.01){lr = learningrate; selectgrads = randgrads;}
    void update_params(LayerDense layer, int i);
    void update_params(Subsetlayer layer, int i);


private:
    double lr;
    std::vector<std::vector<std::tuple<int, int>>> selectgrads;
};


void Fewopt::update_params(LayerDense layer, int i)
{
    for(const auto& [row, col] : selectgrads[i])
    {
        MAT_AT(layer.weights, row, col) -= lr * MAT_AT(layer.dweights, row, col);
    }
    
}
void Fewopt::update_params(Subsetlayer layer, int i)
{
    for(const auto& [row, col] : selectgrads[i])
    {
        MAT_AT(layer.weights, row, col) -= lr * MAT_AT(layer.dweights, row, col);
    }
    
}

//need to create this......................!!!!!!!!!!
std::vector<std::tuple<int, int>> getrandweightlabels(double perc, int num_inputs, int num_out)
{
    std::vector<std::tuple<int, int>> randvecs;
    int numweightschange = int(perc * (double)num_inputs * (double)num_out);
    for (size_t i = 0; i < numweightschange; i++)
    {
        int row = rand() % num_inputs;
        int col = rand() % num_out;
        randvecs.emplace_back(row, col);
    }
    //when using in video, add this later to show speed up...
    //..................
    std::sort(randvecs.begin(), randvecs.end(),
        [](const auto& a, const auto& b) {
            // Compare the first element (index 0) of the two tuples
            return std::get<0>(a) < std::get<0>(b);
        }
    );
    //..................

    return randvecs;
}


int main()
{
    srand(37);
    std::vector<std::vector<float>> images = read_images(imagesfilename);
    std::vector<std::vector<int>> labels = read_labels(labelsfilename);
    
    Mat trainimages = mat_alloc(num_samples, n_inputs);
    Mat trainlabels = mat_alloc(num_samples, num_classes);
    std::vector<std::tuple<int, int>> randweilabels1;
    std::vector<std::tuple<int, int>> randweilabels2;
    std::vector<std::tuple<int, int>> randweilabels3;
    
   
    randweilabels1 = getrandweightlabels(percchangeableweights, n_inputs, n_neurons);
    randweilabels2 = getrandweightlabels(percchangeableweights, n_neurons, n_neurons);
    randweilabels3 = getrandweightlabels(percchangeableweights, n_neurons, num_classes);
   
     std::vector<std::vector<std::tuple<int, int>>> randweilabels;
    randweilabels.push_back(randweilabels1);
    randweilabels.push_back(randweilabels2);
    randweilabels.push_back(randweilabels3);

    Subsetlayer ld1(n_inputs, n_neurons, false, num_samples, randweilabels[0]);
    Relu_Activation relu(num_samples, n_neurons);
    LayerDense ld2(n_neurons, n_neurons, false, num_samples);
    Relu_Activation relu2(num_samples, n_neurons);
    LayerDense ld3(n_neurons, num_classes, false, num_samples);
    
    Activation_softmax soft(num_samples, num_classes);
    Loss_categoricalCrossentropy loss(num_classes, num_samples);

   
    
    Fewopt opt(randweilabels, lr);
    std::vector<int> randvec;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < num_iters; ++j)
    {
        randvec = getrandvec(num_samples);
        trainimages = randomizeImages(images, randvec);
        trainlabels = randomizeLabels(labels, randvec);
        
        ld1.forward(trainimages);
        relu.forward(ld1.output);
        ld2.forward(relu.output);
        relu2.forward(ld2.output);
        ld3.forward(relu2.output);
        soft.forward(ld3.output);
        // mat_print(soft.output);
        // std::cout << "shape " << ld3.output.rows << ", " << ld3.output.cols << std::endl;
        loss.forward(soft.output, trainlabels);
        
        
        loss.backward(soft.output, trainlabels);
        if (j% check_iter ==0 )
        {
            printf("iter: %d\n", j);
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
        ld1.backward(relu.dinputs);



        opt.update_params(ld3, 2);
        opt.update_params(ld2, 1);
        opt.update_params(ld1, 0);
        //replace these.......

        // opt.update_params(ld3);
        // opt.update_params(ld2);
        // opt.update_params(ld1);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = stop - start;

    std::cout << "time " << duration.count()/1000000 << std::endl;
    
    randvec = getrandvec(num_samples);
    trainimages = randomizeImages(images, randvec);
    trainlabels = randomizeLabels(labels, randvec);
    
    ld1.forward(trainimages);
    relu.forward(ld1.output);
    ld2.forward(relu.output);
    relu2.forward(ld2.output);
    ld3.forward(relu2.output);
    soft.forward(ld3.output);
    std::cout << "preds\n";
    mat_print(soft.output);
    std::cout << "yvals\n";
    mat_print(trainlabels);
    return 0;
}
