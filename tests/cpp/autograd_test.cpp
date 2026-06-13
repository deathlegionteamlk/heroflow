#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace heroflow;

void test_autograd_basic() {
    Tensor a({2.0f}, {1});
    a.set_requires_grad(true);
    Tensor b({3.0f}, {1});
    b.set_requires_grad(true);
    
    Tensor c = mul(a, b); // c = a * b
    c.backward();
    
    assert(a.grad().data()[0] == 3.0f);
    assert(b.grad().data()[0] == 2.0f);
    
    std::cout << "test_autograd_basic passed" << std::endl;
}

void test_autograd_complex() {
    // y = a * b + c
    Tensor a({2.0f}, {1}); a.set_requires_grad(true);
    Tensor b({3.0f}, {1}); b.set_requires_grad(true);
    Tensor c({4.0f}, {1}); c.set_requires_grad(true);
    
    Tensor y = add(mul(a, b), c);
    y.backward();
    
    assert(a.grad().data()[0] == 3.0f);
    assert(b.grad().data()[0] == 2.0f);
    assert(c.grad().data()[0] == 1.0f);
    
    std::cout << "test_autograd_complex passed" << std::endl;
}

int main() {
    test_autograd_basic();
    test_autograd_complex();
    return 0;
}
