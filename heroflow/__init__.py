from . import heroflow_internal as _hi
import numpy as np

class Tensor:
    def __init__(self, data, device=_hi.Device.CPU, requires_grad=False):
        if isinstance(data, (list, tuple, np.ndarray)):
            self._tensor = _hi.Tensor(np.array(data, dtype=np.float32), device)
        elif isinstance(data, _hi.Tensor):
            self._tensor = data
        elif isinstance(data, (float, int, np.float32, np.int32, np.float64, np.int64)):
            self._tensor = _hi.Tensor(np.array([data], dtype=np.float32), device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if requires_grad:
            self._tensor.requires_grad = True

    @property
    def shape(self):
        return self._tensor.shape()

    @property
    def requires_grad(self):
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._tensor.requires_grad = value

    def backward(self):
        self._tensor.backward()

    @property
    def grad(self):
        g = self._tensor.grad()
        if g.size() == 0:
            return None
        return Tensor(g)

    def numpy(self):
        return self._tensor.data()

    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return Tensor(_hi.add(self._tensor, other._tensor))

    def __radd__(self, other):
        return Tensor(other) + self

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return Tensor(_hi.sub(self._tensor, other._tensor))

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return Tensor(_hi.mul(self._tensor, other._tensor))

    def __rmul__(self, other):
        return Tensor(other) * self

    def __matmul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return Tensor(_hi.matmul(self._tensor, other._tensor))

    def reshape(self, shape):
        self._tensor.reshape(shape)
        return self

    def transpose(self):
        return Tensor(self._tensor.transpose())

    def __repr__(self):
        return f"heroflow.Tensor({self.numpy()}, shape={self.shape}, requires_grad={self.requires_grad})"

def tensor(data, device=_hi.Device.CPU, requires_grad=False):
    return Tensor(data, device, requires_grad)

def relu(x):
    return Tensor(_hi.relu(x._tensor))

def sum(x):
    return Tensor(_hi.sum(x._tensor))
