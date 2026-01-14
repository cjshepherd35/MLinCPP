//embedding table is in gemini conversation.  makes a random one. not sure thats what i want. 
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <vector>

struct twovec
{
    std::vector<int> first;  
    std::vector<int> second;
};


 twovec get_data()
{
    std::ifstream file("input.txt");  

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file!" << std::endl;
    }
    std::string line, text;
    while (std::getline(file, line)) 
    {
        text.append(line);
    }
    file.close();

    std::vector<int> numtext;
    for(char val: text)
    {
        numtext.emplace_back((int)val);
    }

    std::set<char> unique_set(text.begin(), text.end());
    
    std::vector<char> sorted_uniquechars(unique_set.begin(), unique_set.end());
    std::vector<int> unique_nums;
    for (char elem : sorted_uniquechars)
    {
        unique_nums.emplace_back((int)elem);
    }
    return {numtext, unique_nums};
}


twovec get_train_data(twovec d)
{
    std::vector<int> x = d.first;
    std::vector<int> vocab = d.second;

    std::vector<int> x_train;
    std::vector<int> x_test;

    int trainsize = 0.9*x.size();
    for (size_t i = 0; i < x.size(); i++)
    {
        if (i < trainsize)
        {
            x_train.emplace_back(x[i]);
        }
        else
        {
            x_test.emplace_back(x[i]);
        }
    }

    return {x_train, x_test};

}
