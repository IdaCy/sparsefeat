import torch
import torch.nn as nn
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_weight=0.01):
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

def preprocess_activations(activations, fixed_size=None):
    """
    Ensures all activations are the same size by truncating or padding.
    """
    processed = []
    for batch in activations:
        for act in batch:
            flat_act = act.view(-1).numpy()  # Flatten the tensor
            if fixed_size:
                # Truncate or pad to the fixed size
                if len(flat_act) > fixed_size:
                    flat_act = flat_act[:fixed_size]
                elif len(flat_act) < fixed_size:
                    flat_act = np.pad(flat_act, (0, fixed_size - len(flat_act)))
            processed.append(flat_act)
    return np.array(processed)

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
    # Load activations saved from the extractor
    activations = torch.load("activations.pt")

    # Determine a fixed size for activations
    fixed_size = 5376  # Set this to the smallest observed size or desired size
    data = preprocess_activations(activations, fixed_size=fixed_size)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Train the sparse autoencoder
    model = train_autoencoder(data, input_size=fixed_size, hidden_size=128)
    torch.save(model, "sparse_autoencoder.pt")
    print("Model saved!")
