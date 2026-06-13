#include "tensor.hpp"
#include <stdexcept>
#include <iostream>

namespace heroflow {

Tensor relu(Tensor x) {
    Tensor result(x.shape(), x.device());
    for (int i = 0; i < x.size(); ++i) {
        result.data()[i] = std::max(0.0f, x.data()[i]);
    }
    if (x.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Node>();
        node->inputs = {x};
        node->backward_fn = [x](const Tensor& grad) {
            Tensor grad_x(x.shape(), x.device());
            for (int i = 0; i < x.size(); ++i) {
                grad_x.data()[i] = x.data()[i] > 0 ? grad.data()[i] : 0.0f;
            }
            return std::vector<Tensor>{grad_x};
        };
        result.set_grad_fn(node);
    }
    return result;
}

Tensor sum(Tensor x) {
    float total = 0.0f;
    for (int i = 0; i < x.size(); ++i) total += x.data()[i];
    Tensor result({total}, {1}, x.device());
    if (x.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Node>();
        node->inputs = {x};
        node->backward_fn = [x](const Tensor& grad) {
            Tensor grad_x(x.shape(), x.device());
            float g = grad.data()[0];
            for (int i = 0; i < x.size(); ++i) grad_x.data()[i] = g;
            return std::vector<Tensor>{grad_x};
        };
        result.set_grad_fn(node);
    }
    return result;
}

// Improved add with basic broadcasting for bias (1, N)
Tensor add(Tensor a, Tensor b) {
    if (a.shape() != b.shape()) {
        // Simple broadcasting: if b is (1, N) and a is (M, N)
        if (b.shape().size() == 2 && b.shape()[0] == 1 && a.shape().size() == 2 && a.shape()[1] == b.shape()[1]) {
            int M = a.shape()[0];
            int N = a.shape()[1];
            Tensor result(a.shape(), a.device());
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    result.data()[i * N + j] = a.data()[i * N + j] + b.data()[j];
                }
            }
            if (a.requires_grad() || b.requires_grad()) {
                result.set_requires_grad(true);
                auto node = std::make_shared<Node>();
                node->inputs = {a, b};
                node->backward_fn = [a, b, M, N](const Tensor& grad) {
                    Tensor grad_a = grad;
                    Tensor grad_b({1, N}, b.device());
                    for (int j = 0; j < N; ++j) {
                        float sum_j = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            sum_j += grad.data()[i * N + j];
                        }
                        grad_b.data()[j] = sum_j;
                    }
                    return std::vector<Tensor>{grad_a, grad_b};
                };
                result.set_grad_fn(node);
            }
            return result;
        }
        throw std::runtime_error("Shapes must match for element-wise addition, or broadcasting failed");
    }
    Tensor result(a.shape(), a.device());
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* res_ptr = result.data();
    for (int i = 0; i < a.size(); ++i) {
        res_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->backward_fn = [](const Tensor& grad) {
            return std::vector<Tensor>{grad, grad};
        };
        result.set_grad_fn(node);
    }

    return result;
}

Tensor sub(Tensor a, Tensor b) {
    if (a.shape() != b.shape()) {
         // Simple broadcasting: if b is (1, N) and a is (M, N)
        if (b.shape().size() == 2 && b.shape()[0] == 1 && a.shape().size() == 2 && a.shape()[1] == b.shape()[1]) {
            int M = a.shape()[0];
            int N = a.shape()[1];
            Tensor result(a.shape(), a.device());
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    result.data()[i * N + j] = a.data()[i * N + j] - b.data()[j];
                }
            }
            if (a.requires_grad() || b.requires_grad()) {
                result.set_requires_grad(true);
                auto node = std::make_shared<Node>();
                node->inputs = {a, b};
                node->backward_fn = [a, b, M, N](const Tensor& grad) {
                    Tensor grad_a = grad;
                    Tensor grad_b({1, N}, b.device());
                    for (int j = 0; j < N; ++j) {
                        float sum_j = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            sum_j -= grad.data()[i * N + j];
                        }
                        grad_b.data()[j] = sum_j;
                    }
                    return std::vector<Tensor>{grad_a, grad_b};
                };
                result.set_grad_fn(node);
            }
            return result;
        }
        throw std::runtime_error("Shapes must match for element-wise subtraction");
    }
    Tensor result(a.shape(), a.device());
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* res_ptr = result.data();
    for (int i = 0; i < a.size(); ++i) {
        res_ptr[i] = a_ptr[i] - b_ptr[i];
    }

    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->backward_fn = [](const Tensor& grad) {
            Tensor neg_grad(grad.shape(), grad.device());
            for (int i = 0; i < grad.size(); ++i) neg_grad.data()[i] = -grad.data()[i];
            return std::vector<Tensor>{grad, neg_grad};
        };
        result.set_grad_fn(node);
    }
    return result;
}

Tensor mul(Tensor a, Tensor b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shapes must match for element-wise multiplication");
    }
    Tensor result(a.shape(), a.device());
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* res_ptr = result.data();
    for (int i = 0; i < a.size(); ++i) {
        res_ptr[i] = a_ptr[i] * b_ptr[i];
    }

    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->backward_fn = [a, b](const Tensor& grad) {
            Tensor grad_a = mul(grad, b);
            Tensor grad_b = mul(grad, a);
            return std::vector<Tensor>{grad_a, grad_b};
        };
        result.set_grad_fn(node);
    }
    return result;
}

Tensor matmul(Tensor a, Tensor b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        // Log shapes to debug
        std::cerr << "matmul shapes: a(";
        for (int s : a.shape()) std::cerr << s << ",";
        std::cerr << ") b(";
        for (int s : b.shape()) std::cerr << s << ",";
        std::cerr << ")" << std::endl;
        throw std::runtime_error("Only 2D matmul is implemented");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Incompatible shapes for matmul");
    }
    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];
    Tensor result({M, N}, a.device());
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* res_ptr = result.data();
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum_val = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum_val += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            res_ptr[i * N + j] = sum_val;
        }
    }

    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->backward_fn = [a, b](const Tensor& grad) {
            // grad_a = grad @ b.T
            // grad_b = a.T @ grad
            Tensor b_t = b.transpose();
            Tensor a_t = a.transpose();
            return std::vector<Tensor>{matmul(grad, b_t), matmul(a_t, grad)};
        };
        result.set_grad_fn(node);
    }
    return result;
}

} // namespace heroflow
