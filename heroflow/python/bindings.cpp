#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.hpp"
#include <iostream>

namespace py = pybind11;
using namespace heroflow;

PYBIND11_MODULE(heroflow_internal, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, Device>(), py::arg("shape"), py::arg("device") = Device::CPU)
        .def(py::init([](py::array_t<float> b, Device device) {
            py::buffer_info info = b.request();
            std::vector<int> shape;
            for (auto s : info.shape) shape.push_back((int)s);
            std::vector<float> data((float*)info.ptr, (float*)info.ptr + info.size);
            return Tensor(data, shape, device);
        }), py::arg("data"), py::arg("device") = Device::CPU)
        .def("shape", [](const Tensor& t) {
            return t.shape();
        })
        .def("size", &Tensor::size)
        .def("data", [](Tensor& t) {
             std::vector<ssize_t> shape;
             for (int s : t.shape()) shape.push_back(s);
             std::vector<ssize_t> strides;
             ssize_t s = sizeof(float);
             for (int i = t.shape().size() - 1; i >= 0; --i) {
                 strides.insert(strides.begin(), s);
                 s *= t.shape()[i];
             }
             if (strides.empty()) strides.push_back(sizeof(float));
             return py::array_t<float>(shape, strides, t.data(), py::cast(t));
        })
        .def("backward", [](Tensor& t) {
             t.backward();
        })
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def("grad", &Tensor::grad)
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose);

    m.def("add", &add);
    m.def("sub", &sub);
    m.def("mul", &mul);
    m.def("matmul", &matmul);
    m.def("relu", &relu);
    m.def("sum", &sum);
}
