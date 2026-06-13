import heroflow

class Module:
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        params = []
        for p in self._parameters.values():
            if isinstance(p, heroflow.Tensor):
                params.append(p)
            elif isinstance(p, Module):
                params.extend(p.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        import numpy as np
        limit = np.sqrt(2.0 / in_features)
        self.weight = heroflow.tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )
        self.bias = heroflow.tensor(
            np.zeros((1, out_features)),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias

    def forward(self, x):
        return (x @ self.weight) + self.bias

def mse_loss(input, target):
    diff = input - target
    sq_diff = diff * diff
    return heroflow.sum(sq_diff)
