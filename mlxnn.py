import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Define a simple neural network (MLP)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.layer2(self.relu(self.layer1(x)))

# Define the loss function
def l2_loss(model, x, y):
    y_hat = model(x)
    return (y_hat - y).square().mean()

# Initialize the model, loss, and optimizer
input_dim = 10
hidden_dim = 64
output_dim = 1

model = MLP(input_dim, hidden_dim, output_dim)
loss_and_grad_fn = nn.value_and_grad(model, l2_loss)
optimizer = optim.Adam(learning_rate=0.01)

# Generate some dummy data
num_samples = 100
X = mx.random.normal([num_samples, input_dim])
y = mx.random.normal([num_samples, output_dim])

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward and backward pass
    loss, grads = loss_and_grad_fn(model, X, y)
    # Compute predictions and accuracy
    predictions = model(X)
    predicted_labels = (predictions > 0.5).astype(mx.float32)
    true_labels = (y > 0.5).astype(mx.float32)
    accuracy = (predicted_labels == true_labels).mean().item()
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")
    # Update model parameters
    optimizer.update(model, grads)

    # Evaluate the updated parameters
    mx.eval(model.parameters(), optimizer.state)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# You can now use the trained model for inference
# For example:
new_data = mx.random.normal([1, input_dim])
prediction = model(new_data)
print(f"Prediction: {prediction}")
