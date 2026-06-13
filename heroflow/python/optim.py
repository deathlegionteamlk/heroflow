class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            g = p.grad
            if g is not None:
                g.numpy().fill(0)

class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            g_tensor = p.grad
            if g_tensor is not None:
                g = g_tensor.numpy()
                if g.shape != p.shape:
                    g = g.reshape(p.shape)
                # Modify p's internal data directly
                p_data = p.numpy()
                p_data -= self.lr * g
