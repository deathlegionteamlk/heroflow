#include "tensor.hpp"
#include <algorithm>
#include <set>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace heroflow {

// Helper to accumulate gradients
void accumulate_grad(Tensor& t, const Tensor& g) {
    if (!t.storage()) return;
    if (t.grad().storage() == nullptr) {
        Tensor new_grad(g.shape(), g.device());
        std::copy(g.data(), g.data() + g.size(), new_grad.data());
        t.set_grad(new_grad);
    } else {
        Tensor current_grad = t.grad();
        for (int i = 0; i < t.size() && i < g.size(); ++i) {
            current_grad.data()[i] += g.data()[i];
        }
    }
}

void Tensor::backward() {
    if (!storage_ || !requires_grad()) return;

    if (grad().storage() == nullptr) {
        Tensor g(shape(), device());
        std::fill(g.data(), g.data() + g.size(), 1.0f);
        set_grad(g);
    }

    std::vector<Tensor> topo;
    std::set<TensorStorage*> visited;
    std::vector<Tensor> stack;
    stack.push_back(*this);

    while(!stack.empty()){
        Tensor curr = stack.back();
        if(visited.find(curr.storage().get()) != visited.end()){
            stack.pop_back();
            continue;
        }
        
        bool all_inputs_visited = true;
        if(curr.grad_fn()){
            for(auto& input : curr.grad_fn()->inputs){
                if(input.storage() && visited.find(input.storage().get()) == visited.end()){
                    stack.push_back(input);
                    all_inputs_visited = false;
                }
            }
        }
        
        if(all_inputs_visited){
            visited.insert(curr.storage().get());
            topo.push_back(curr);
            stack.pop_back();
        }
    }

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor t = *it;
        if (t.grad_fn()) {
            auto current_grad = t.grad();
            if (current_grad.storage() == nullptr) continue;
            
            auto grads = t.grad_fn()->backward_fn(current_grad);
            for (size_t i = 0; i < grads.size(); ++i) {
                if (i < t.grad_fn()->inputs.size() && t.grad_fn()->inputs[i].requires_grad()) {
                    accumulate_grad(t.grad_fn()->inputs[i], grads[i]);
                }
            }
        }
    }
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    if (!storage_) return;
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != size()) {
        throw std::runtime_error("New shape size must match original size");
    }
    storage_->shape = new_shape;
}

Tensor Tensor::transpose() const {
    if (!storage_) return Tensor();
    if (shape().size() != 2) {
        throw std::runtime_error("Only 2D transpose is implemented");
    }
    int rows = shape()[0];
    int cols = shape()[1];
    Tensor result({cols, rows}, device());
    const float* src = data();
    float* dst = result.data();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return result;
}

} // namespace heroflow
