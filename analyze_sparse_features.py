import torch
import numpy as np
import matplotlib.pyplot as plt
from train_sparse_autoencoder import SparseAutoencoder

def preprocess_activations(activations, fixed_size):
    flattened = []
    for batch in activations:
        for layer in batch:
            layer_flat = layer.view(-1).numpy()
            if len(layer_flat) > fixed_size:
                # Truncate if larger
                layer_flat = layer_flat[:fixed_size]
            else:
                # Pad with zeros if smaller
                layer_flat = np.pad(layer_flat, (0, fixed_size - len(layer_flat)))
            flattened.append(layer_flat)
    return np.array(flattened)

def analyze_features(model_file="sparse_autoencoder.pt", activations_file="activations.pt", fixed_size=5376):
    model = torch.load(model_file)
    activations = torch.load(activations_file)

    # Preprocess activations
    data = preprocess_activations(activations, fixed_size=fixed_size)
    data = torch.tensor(data, dtype=torch.float32)

    # Encode data with the sparse autoencoder
    encoded, _ = model(data)

    # Plot histogram of encoded sparse features
    plt.hist(encoded.detach().numpy().flatten(), bins=50)
    plt.title("Histogram of Encoded Sparse Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    analyze_features()
