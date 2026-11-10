
import torch
from models import FedANN, FedConvNN

torch.manual_seed(42)

# Create eight dummy samples
x = torch.randn(8, 561)

# Create both models
ann = FedANN()
cnn = FedConvNN()

# Evaluation mode disables dropout
ann.eval()
cnn.eval()

with torch.no_grad():
    ann_out = ann(x)
    cnn_out = cnn(x)

# Count trainable parameters
ann_params = sum(p.numel() for p in ann.parameters())
cnn_params = sum(p.numel() for p in cnn.parameters())

print("Input shape:", x.shape)

print("\nFedANN")
print("Output:", ann_out.shape)
print("Parameters:", ann_params)

print("\nFedConvNN")
print("Output:", cnn_out.shape)
print("Parameters:", cnn_params)

assert ann_out.shape == (8, 6)
assert cnn_out.shape == (8, 6)

print("\nBoth models working successfully!")
