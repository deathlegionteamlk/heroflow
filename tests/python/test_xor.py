import heroflow
from heroflow.python.nn import Linear, mse_loss
from heroflow.python.optim import SGD
import numpy as np

def test_xor():
    # XOR is not linearly separable, but let's see if we can train a small net
    # We need a hidden layer and ReLU
    class Net(heroflow.python.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(2, 4)
            self.fc2 = Linear(4, 1)
            self._parameters['fc1'] = self.fc1
            self._parameters['fc2'] = self.fc2

        def forward(self, x):
            x = heroflow.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    X = heroflow.tensor([[0,0], [0,1], [1,0], [1,1]])
    Y = heroflow.tensor([[0], [1], [1], [0]])
    
    model = Net()
    optimizer = SGD(model.parameters(), lr=0.1)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = mse_loss(output, Y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()[0]}")
            
    print("XOR Predictions:")
    print(model(X).numpy())
    print("XOR test finished")

if __name__ == "__main__":
    test_xor()
