#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace heroflow;

void test_tensor_creation() {
    std::vector<int> shape = {2, 3};
    Tensor t(shape);
    assert(t.shape() == shape);
    assert(t.size() == 6);
    std::cout << "test_tensor_creation passed" << std::endl;
}

void test_tensor_ops() {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    
    Tensor c = add(a, b);
    assert(c.data()[0] == 6.0f);
    assert(c.data()[3] == 12.0f);

    Tensor d = matmul(a, b);
    // [1 2] [5 6]   [19 22]
    // [3 4] [7 8] = [43 50]
    assert(d.data()[0] == 19.0f);
    assert(d.data()[1] == 22.0f);
    assert(d.data()[2] == 43.0f);
    assert(d.data()[3] == 50.0f);
    
    std::cout << "test_tensor_ops passed" << std::endl;
}

int main() {
    // We need a dummy autograd.cpp for linking, but for now we'll just test Tensor/Ops
    test_tensor_creation();
    test_tensor_ops();
    return 0;
}
