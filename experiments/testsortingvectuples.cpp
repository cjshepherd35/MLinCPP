#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <string>

// Helper to print the tuple
void print_tuple(const std::tuple<int, std::string, double>& t) {
    std::cout << "(" << std::get<0>(t) << ", "
              << std::get<1>(t) << ", "
              << std::get<2>(t) << ")\n";
}

int main() {
    std::vector<std::tuple<int, std::string, double>> data = {
        {3, "apple", 1.5},
        {2, "banana", 0.7},
        {1, "cherry", 2.2},
        {4, "date", 0.9}
    };

    // --- Sort by the key at index 0 (the 'int') ---

    // The lambda takes two tuples (a and b) and returns true if a < b
    std::sort(data.begin(), data.end(),
        [](const auto& a, const auto& b) {
            // Compare the first element (index 0) of the two tuples
            return std::get<0>(a) < std::get<0>(b);
        }
    );

    std::cout << "Sorted by Key (Index 0):\n";
    for (const auto& t : data) {
        print_tuple(t);
    }
    // Output will be: (1, banana, 0.7), (2, cherry, 2.2), (3, apple, 1.5), (4, date, 0.9)
    
    return 0;
}