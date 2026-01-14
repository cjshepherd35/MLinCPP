//come back to this, not sure why its not working. 
#include <iostream>
#include "matrixstuff.cpp"

int main()
{
    Mat a = mat_alloc(4,2);
    Mat b = mat_alloc(2,3);
    Mat c = mat_alloc(6,4); //a_cols x b_cols, a_rows
    
    double f = 1;
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            MAT_AT(a, i, j) = f;
            f++;
        }
    }
    double k = 1;
    for (size_t i = 0; i < b.rows; i++)
    {
        for (size_t j = 0; j < b.cols; j++)
        {
            MAT_AT(b,i,j) = k;
            k++;
        }
    }
    
    rowtocolkron(c, a, b);
    std::cout << "a \n";
    mat_print(a);
    std::cout << "b \n";
    mat_print(b);
    std::cout << "c \n";
    mat_print(c);

    return 0;
}