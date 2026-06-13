import heroflow
from heroflow.python.nn import Linear, mse_loss
from heroflow.python.optim import SGD
import numpy as np

def test_linear_regression():
    # Synthetic data: y = 2x + 1
    np.random.seed(42)
    X = np.random.rand(10, 1).astype(np.float32)
    Y = 2 * X + 1
    
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.01) # Lower learning rate
    
    X_tensor = heroflow.tensor(X)
    Y_tensor = heroflow.tensor(Y)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = mse_loss(output, Y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()[0]}")
            
    print(f"Final weight: {model.weight.numpy()}")
    print(f"Final bias: {model.bias.numpy()}")
    
    w = model.weight.numpy()[0, 0]
    b = model.bias.numpy()[0, 0]
    print(f"w: {w}, b: {b}")
    assert abs(w - 2.0) < 0.1
    assert abs(b - 1.0) < 0.1
    print("Linear regression test passed!")

if __name__ == "__main__":
    test_linear_regression()
