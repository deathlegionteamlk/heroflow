import heroflow
from heroflow.python.nn import Linear, mse_loss
from heroflow.python.optim import SGD
import numpy as np

def test_debug():
    X = heroflow.tensor([[0.5]])
    W = heroflow.tensor([[2.0]], requires_grad=True)
    B = heroflow.tensor([[1.0]], requires_grad=True)
    
    out1 = X @ W
    out2 = out1 + B
    Y = heroflow.tensor([[2.5]])
    diff = out2 - Y
    sq = diff * diff
    loss = heroflow.sum(sq)
    loss.backward()
    
    optimizer = SGD([W, B], lr=0.1)
    p_data = W.numpy()
    print(f"p_data type: {type(p_data)}")
    print(f"p_data before: {p_data}")
    p_data -= 0.1 * W.grad.numpy()
    print(f"p_data after: {p_data}")
    print(f"W.numpy() after: {W.numpy()}")

if __name__ == "__main__":
    test_debug()
