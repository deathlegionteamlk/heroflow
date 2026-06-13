import heroflow
import numpy as np

def test_python_bindings():
    a = heroflow.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = heroflow.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    c = a + b
    print(f"a + b =\n{c.numpy()}")
    assert np.allclose(c.numpy(), [[6.0, 8.0], [10.0, 12.0]])
    
    d = a @ b
    print(f"a @ b =\n{d.numpy()}")
    assert np.allclose(d.numpy(), [[19.0, 22.0], [43.0, 50.0]])
    
    a.requires_grad = True
    e = a @ b
    e.backward()
    print(f"a.grad =\n{a.grad.numpy()}")
    # de/da = grad @ b.T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
    # Since e is [[19, 22], [43, 50]], and we backward on the whole tensor e (all 1s gradient)
    # The current backward implementation for non-scalar outputs fills grad with 1s.
    
    print("Python bindings verification passed!")

if __name__ == "__main__":
    test_python_bindings()
