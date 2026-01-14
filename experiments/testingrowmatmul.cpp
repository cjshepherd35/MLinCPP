#include <iostream>
#include "matrixstuff.cpp"
#include  <chrono>

int main()
{
    Mat a = mat_alloc(3,1024);
    Mat b = mat_alloc(1024,4);
    Mat brow = mat_alloc(4,1024);
    Mat dst1 = mat_alloc(3,4);
    Mat dst2 = mat_alloc(3,4);

    mat_rand(a, 0,1.f);
    mat_rand(b, 0, 1.f);
    // mat_fill(a, 1);
    // mat_fill(b, 2);
    mat_transpose(brow, b);
    auto start = std::chrono::high_resolution_clock::now();
    mat_dot(dst1, a, b);
    auto stop = std::chrono::high_resolution_clock::now();
    auto  duration = stop - start;
    std::cout << "dur1: " << duration.count() << std::endl;
    start = std::chrono::high_resolution_clock::now();
    rowwise_mat_dot(dst2, a, brow);
    stop = std::chrono::high_resolution_clock::now();
    duration = stop - start;
    std::cout << "dur2: " << duration.count() << std::endl;

    // std::cout << "dst1\n";
    // mat_print(dst1);
    // std::cout  << "dst2\n";
    // mat_print(dst2);
    return 0;
}