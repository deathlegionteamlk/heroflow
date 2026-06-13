#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace heroflow {

enum class Device { CPU, GPU };

class Tensor;

struct Node {
    std::vector<Tensor> inputs;
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn;
};

struct TensorStorage {
    std::vector<float> data;
    std::vector<int> shape;
    Device device;
    
    bool requires_grad = false;
    std::shared_ptr<TensorStorage> grad = nullptr;
    std::shared_ptr<Node> grad_fn = nullptr;

    TensorStorage(const std::vector<int>& s, Device d) : shape(s), device(d) {
        int size = std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
        data.resize(size, 0.0f);
    }
    
    TensorStorage(const std::vector<float>& d, const std::vector<int>& s, Device dev) 
        : data(d), shape(s), device(dev) {}
};

class Tensor {
public:
    Tensor() : storage_(nullptr) {}
    Tensor(std::shared_ptr<TensorStorage> storage) : storage_(storage) {}
    Tensor(const std::vector<int>& shape, Device device = Device::CPU) 
        : storage_(std::make_shared<TensorStorage>(shape, device)) {}
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, Device device = Device::CPU)
        : storage_(std::make_shared<TensorStorage>(data, shape, device)) {}

    const std::vector<int>& shape() const { 
        static std::vector<int> empty;
        return storage_ ? storage_->shape : empty; 
    }
    int size() const { return storage_ ? storage_->data.size() : 0; }
    Device device() const { return storage_ ? storage_->device : Device::CPU; }
    
    float* data() { return storage_ ? storage_->data.data() : nullptr; }
    const float* data() const { return storage_ ? storage_->data.data() : nullptr; }

    bool requires_grad() const { return storage_ ? storage_->requires_grad : false; }
    void set_requires_grad(bool r) { if(storage_) storage_->requires_grad = r; }

    Tensor grad() const { 
        if (!storage_ || !storage_->grad) return Tensor();
        return Tensor(storage_->grad); 
    }
    void set_grad(Tensor g) { if(storage_) storage_->grad = g.storage_; }

    std::shared_ptr<Node> grad_fn() const { return storage_ ? storage_->grad_fn : nullptr; }
    void set_grad_fn(std::shared_ptr<Node> fn) { if(storage_) storage_->grad_fn = fn; }

    void backward();

    void reshape(const std::vector<int>& new_shape);
    Tensor transpose() const;

    std::shared_ptr<TensorStorage> storage() const { return storage_; }

private:
    std::shared_ptr<TensorStorage> storage_;
};

Tensor add(Tensor a, Tensor b);
Tensor sub(Tensor a, Tensor b);
Tensor mul(Tensor a, Tensor b);
Tensor matmul(Tensor a, Tensor b);
Tensor relu(Tensor x);
Tensor sum(Tensor x);

} // namespace heroflow
