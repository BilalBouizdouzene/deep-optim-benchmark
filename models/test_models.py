import torch
from models.mlp import DeepMLP
from models.cnn import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test MLP
mlp = DeepMLP().to(device)
x_mlp = torch.randn(16, 1, 28, 28).to(device)
out_mlp = mlp(x_mlp)
print("MLP output shape:", out_mlp.shape)  # (16, 10)

# Test CNN
cnn = SimpleCNN().to(device)
x_cnn = torch.randn(16, 1, 28, 28).to(device)
out_cnn = cnn(x_cnn)
print("CNN output shape:", out_cnn.shape)  # (16, 10)
