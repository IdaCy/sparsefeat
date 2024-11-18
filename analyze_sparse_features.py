import torch
import matplotlib.pyplot as plt
from train_sparse_autoencoder import SparseAutoencoder

def analyze_features(model_file="sparse_autoencoder.pt", activations_file="activations.pt"):
    model = torch.load(model_file)
    activations = torch.load(activations_file)
    flattened = [act.view(-1).numpy() for batch in activations for act in batch]
    data = torch.tensor(flattened, dtype=torch.float32)
    
    encoded, _ = model(data)
    plt.hist(encoded.detach().numpy().flatten(), bins=50)
    plt.title("Histogram of Encoded Sparse Features")
    plt.show()

if __name__ == "__main__":
    analyze_features()
