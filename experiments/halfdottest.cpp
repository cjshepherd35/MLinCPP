#include <iostream>
#include "../nnfromscratchfrompyversion.cpp"


int main()
{
    Mat a = mat_alloc(2,2);
    mat_fill(a, 1.0);
    LayerDense ld(2, 4, false, 2);
    ld.halfforward(a);
    mat_print(ld.duboutput);
    ld.halfbackward(ld.output);


    return 0;
}