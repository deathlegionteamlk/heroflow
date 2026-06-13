<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=7,13,20&height=220&section=header&text=heroflow&fontSize=85&fontColor=ffffff&fontAlignY=38&desc=C%2B%2B%20ML%20engine.%20Python%20API.%20Autograd%20built%20in.&descAlignY=60&descSize=20&animation=fadeIn" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2800&pause=900&color=8B5CF6&center=true&vCenter=true&multiline=true&width=680&height=80&lines=C%2B%2B+core+for+speed.+Python+API+for+sanity.;Autograd+that+works+like+you+expect.;Research+to+deployment%2C+one+framework." alt="Typing animation"/>

<br/><br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![C++](https://img.shields.io/badge/Core-C%2B%2B17-ef4444?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org)
[![GPU](https://img.shields.io/badge/GPU-Planned-f59e0b?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/deathlegionteam/heroflow)
[![Built by](https://img.shields.io/badge/💀-Death%20Legion%20Team-1a1a1a?style=for-the-badge)](https://github.com/deathlegionteam)

</div>

---

## ⚡ What is heroflow?

Python is the language of machine learning, but Python is slow. The standard fix is to write the hot paths in C++ and expose them to Python — which is exactly what PyTorch, TensorFlow, and JAX all do.

heroflow takes the same approach. The tensor engine, memory management, and gradient computation are written in C++. The Python layer sits on top and gives you the familiar API: define a model, run a forward pass, call `.backward()`, step the optimizer. The performance-critical work happens below in compiled code.

It's a from-scratch implementation — not a wrapper around another framework — which means the internals are readable, extensible, and not carrying fifteen years of accumulated complexity.

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="600"/>
</div>

---

## ✨ Features

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="80"/>
</div>

<table>
<tr>
<td width="50%">

### 🔴 C++ Core
The tensor engine is written in C++17. No Python overhead in the compute path — allocations, arithmetic, and gradient tracking all happen in compiled code.

### 🐍 Python API
The Python layer follows conventions you already know: `tensor()`, `nn.Linear`, `optim.SGD`, `.backward()`. If you've used PyTorch, reading heroflow code requires no mental translation.

### 🧠 Automatic Differentiation
The autograd engine builds a computation graph as your forward pass runs. Call `.backward()` on any scalar and gradients flow back through every operation that produced it.

</td>
<td width="50%">

### 📐 Tensor Operations
Standard tensor arithmetic — addition, multiplication, matmul, reshape, slicing — all tracked by autograd so gradients work through any combination of them.

### 🔗 Full ML Workflow
Layers, loss functions, and optimizers are built in. `heroflow.python.nn` and `heroflow.python.optim` give you what you need to define, train, and evaluate models without reaching for another library.

### 🚀 GPU Acceleration *(planned)*
CUDA support is on the roadmap. The C++ core is designed with device abstraction so the GPU backend can be added without rewriting the Python API.

</td>
</tr>
</table>

---

## 🏗️ Architecture

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif" width="350"/>
</div>

```
┌──────────────────────────────────────────────────────────┐
│                     heroflow stack                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  Python API Layer                  │  │
│  │   heroflow.tensor · nn.Linear · optim.SGD          │  │
│  │   loss functions · .backward() · .numpy()          │  │
│  └────────────────────────┬───────────────────────────┘  │
│                           │  pybind11 bindings           │
│  ┌────────────────────────▼───────────────────────────┐  │
│  │                  C++ Core Engine                   │  │
│  │   Tensor storage · Memory management               │  │
│  │   Autograd graph · Gradient computation            │  │
│  │   CPU dispatch (GPU dispatch planned)              │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

Python talks to C++ through pybind11 bindings. The C++ layer owns all tensor memory and the computation graph. The Python layer is a thin convenience wrapper — there's no tensor data living in Python objects.

---

## 📦 Install

```bash
git clone https://github.com/deathlegionteam/heroflow.git
cd heroflow
pip install -e .
```

The build step compiles the C++ core. You'll need a C++17-capable compiler (GCC 9+ or Clang 10+) and CMake. On most systems:

```bash
# Linux / macOS
sudo apt install build-essential cmake    # Ubuntu
brew install cmake                        # macOS

# Then install
pip install -e .
```

---

## 🚀 Quickstart

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="80"/>
</div>

### Train a linear model

```python
import heroflow
from heroflow.python.nn import Linear, mse_loss
from heroflow.python.optim import SGD

# Data: predict y = 4x
X = heroflow.tensor([[0.5], [1.0], [1.5], [2.0]])
Y = heroflow.tensor([[2.0], [4.0], [6.0], [8.0]])

model     = Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    output = model(X)
    loss   = mse_loss(output, Y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.numpy():.6f}")
```

---

### Multi-layer network

Stack layers to build a deeper model:

```python
import heroflow
from heroflow.python.nn import Linear, ReLU, mse_loss, Sequential
from heroflow.python.optim import SGD

model = Sequential([
    Linear(2, 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 1),
])

optimizer = SGD(model.parameters(), lr=0.01)

X = heroflow.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Y = heroflow.tensor([[3.0], [7.0], [11.0]])

for epoch in range(300):
    optimizer.zero_grad()
    output = model(X)
    loss   = mse_loss(output, Y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.numpy():.6f}")
```

---

### Autograd directly

You don't need a model to use autograd. Any tensor computation is differentiable:

```python
import heroflow

x = heroflow.tensor([[2.0]], requires_grad=True)
w = heroflow.tensor([[3.0]], requires_grad=True)
b = heroflow.tensor([[1.0]], requires_grad=True)

# y = wx + b
y = w * x + b   # → 7.0

y.backward()

print(x.grad)   # dy/dx = w = 3.0
print(w.grad)   # dy/dw = x = 2.0
print(b.grad)   # dy/db = 1.0
```

---

## 📋 API reference

### Tensors

| Call | Description |
|---|---|
| `heroflow.tensor(data, requires_grad=False)` | Create a tensor from a Python list or numpy array |
| `.backward()` | Backpropagate gradients from this tensor |
| `.numpy()` | Convert to a NumPy array |
| `.grad` | Gradient after `.backward()` |
| `.shape` | Tuple of dimensions |
| `.zero_grad()` | Clear accumulated gradient |

### `heroflow.python.nn`

| Module | Description |
|---|---|
| `Linear(in_features, out_features)` | Fully connected layer |
| `ReLU()` | Rectified linear activation |
| `Sigmoid()` | Sigmoid activation |
| `Sequential(layers)` | Chain of layers |
| `mse_loss(output, target)` | Mean squared error |

### `heroflow.python.optim`

| Optimizer | Description |
|---|---|
| `SGD(params, lr, momentum=0)` | Stochastic gradient descent |
| `.zero_grad()` | Zero out accumulated gradients |
| `.step()` | Apply one gradient update |

---

## 🗺️ Roadmap

<table>
<tr>
<td width="33%">

**Now ✅**
- C++ tensor engine
- Autograd
- Linear layers
- ReLU, Sigmoid
- MSE loss
- SGD optimizer
- Python bindings

</td>
<td width="33%">

**Next 🔄**
- Adam optimizer
- Convolutional layers
- Cross-entropy loss
- Model save/load
- NumPy interop improvements

</td>
<td width="33%">

**Planned 📋**
- CUDA GPU support
- Distributed training
- ONNX export
- Benchmark suite
- Extended layer library

</td>
</tr>
</table>

---

## 🤝 Contributing

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="400"/>
</div>

The C++ core and Python bindings are both fair game for contributions. New ops, new layers, optimizer implementations, and tests are all useful. Open an issue first for anything that changes the public API.

```bash
git clone https://github.com/deathlegionteam/heroflow.git
cd heroflow
pip install -e ".[dev]"
pytest
```

---

## 🛡️ License

MIT © [Death Legion Team](https://github.com/deathlegionteam)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=7,13,20&height=100&section=footer&animation=fadeIn" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=1000&color=8B5CF6&center=true&vCenter=true&width=540&lines=C%2B%2B+speed.+Python+convenience.+No+compromise.;Autograd+from+scratch.+Not+a+wrapper.;💀+Built+by+Death+Legion+Team." alt="Footer typing"/>

</div>
