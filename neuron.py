import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.last_input = None
        self.last_z = None
        self.last_output = None

    def forward(self, x):
        self.last_input = x
        z = np.dot(x, self.weights) + self.bias
        self.last_z = z
        self.last_output = self._activate(z)
        return self.last_output

    def _activate(self, z):
        # Using sigmoid activation
        return 1 / (1 + np.exp(-z))

    def _activate_deriv(self, z):
        sig = self._activate(z)
        return sig * (1 - sig)

    def backward(self, dL_dout, learning_rate):
        dz = dL_dout * self._activate_deriv(self.last_z)
        dL_dw = dz * self.last_input
        dL_db = dz
        # Update weights and bias
        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
        # Return gradient for previous layer
        return dz * self.weights

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # hidden_sizes: list of 3 integers for 3 hidden layers
        assert len(hidden_sizes) == 3, "Must provide 3 hidden layer sizes"
        self.layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            self.layers.append([Neuron(prev_size) for _ in range(h_size)])
            prev_size = h_size
        # Output layer
        self.output_layer = [Neuron(prev_size) for _ in range(output_size)]

    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            x = np.array([neuron.forward(x) for neuron in layer])
            activations.append(x)
        output = np.array([neuron.forward(x) for neuron in self.output_layer])
        activations.append(output)
        return output, activations

    def backward(self, x, y_true, activations, learning_rate):
        # Compute output error (mean squared error loss)
        y_pred = activations[-1]
        dL_dout = 2 * (y_pred - y_true) / y_true.size
        # Output layer
        dL_dprev = np.zeros_like(activations[-2], dtype=np.float64)
        for i, neuron in enumerate(self.output_layer):
            grad = neuron.backward(dL_dout[i], learning_rate)
            dL_dprev += grad
        # Hidden layers (reverse order)
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            prev_activation = activations[l]
            dL_dnext = np.zeros_like(prev_activation, dtype=np.float64)
            for i, neuron in enumerate(layer):
                grad = neuron.backward(dL_dprev[i], learning_rate)
                dL_dnext += grad
            dL_dprev = dL_dnext

    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            loss = 0
            correct = 0
            for x, y in zip(X, Y):
                y_pred, activations = self.forward(x)
                loss += np.mean((y_pred - y) ** 2)
                self.backward(x, y, activations, learning_rate)
                # For accuracy: compare argmax of prediction and label
                if np.argmax(y_pred) == np.argmax(y):
                    correct += 1
            avg_loss = loss / len(X)
            accuracy = correct / len(X)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

# Example usage:
# Create sample data (XOR-like problem for demonstration)
X = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
])
Y = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1],
])

nn = NeuralNetwork(input_size=4, hidden_sizes=[8, 6, 4], output_size=2)
nn.train(X, Y, epochs=10000, learning_rate=0.1)

# Test prediction after training
for x in X:
    output, _ = nn.forward(x)
    print(f"Input: {x}, Output: {output}")