#include <iostream>
#include "../matmul.cpp"

int main()
{
   
    Mat a(3,4);
    Mat b(4,4);

    for (size_t i = 0; i < a.get_rows(); i++)
    {
        for (size_t j = 0; j < a.get_cols(); j++)
        {
            a.set_entry(i,j, (float)i+j);
        }
    }
    for (size_t i = 0; i < b.get_rows(); i++)
    {
        for (size_t j = 0; j < b.get_cols(); j++)
        {
            b.set_entry(i,j,(float)2*(i+j));
        }
        
    }
    
    a.printmat();
    std::cout << std::endl;
    b.printmat();
    std::cout << std::endl;

    Mat c = a.selfAttentiondot(b, 2);

    c.printmat();

    return 0;
}