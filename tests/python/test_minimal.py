import heroflow
from heroflow.python.nn import Linear, mse_loss
from heroflow.python.optim import SGD
import numpy as np

def test_minimal_matmul():
    a = heroflow.tensor([[1.0]], requires_grad=True)
    b = heroflow.tensor([[2.0]], requires_grad=True)
    print("Pre-matmul")
    c = a @ b
    print(f"Post-matmul: {c}")
    print("Pre-backward")
    c.backward()
    print("Post-backward")
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")

if __name__ == "__main__":
    test_minimal_matmul()
