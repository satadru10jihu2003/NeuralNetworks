import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1. / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        hs = []
        for x in inputs:
            x = x.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hs.append(h)
        y = np.dot(self.Why, h) + self.by
        return y, hs

# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    rnn = SimpleRNN(input_size=3, hidden_size=5, output_size=2)
    # Sequence of 4 time steps, each with 3 features
    inputs = [np.random.randn(3) for _ in range(4)]
    output, hidden_states = rnn.forward(inputs)
    print("Output:\n", output)