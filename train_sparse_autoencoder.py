import torch
import torch.nn as nn
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_weight=0.1):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def sparsity_loss(self, encoded):
        sparsity_penalty = torch.mean(torch.abs(encoded))  # L1 regularization
        return self.sparsity_weight * sparsity_penalty

def train_autoencoder(data, input_size, hidden_size, sparsity_weight=0.1, epochs=10, batch_size=32):
    model = SparseAutoencoder(input_size, hidden_size, sparsity_weight)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32)
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch) + model.sparsity_loss(encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data)}")

    return model

if __name__ == "__main__":
    activations = torch.load("activations.pt")
    flattened = [act.view(-1).numpy() for batch in activations for act in batch]  # Flatten activations
    data = np.vstack(flattened)

    model = train_autoencoder(data, input_size=data.shape[1], hidden_size=128)
    torch.save(model, "sparse_autoencoder.pt")
    print("Model saved!")


