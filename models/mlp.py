import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[512, 256, 128, 64, 32], num_classes=10):
        super(DeepMLP, self).__init__()
        layers = []
        in_features = input_size
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # régularisation
            in_features = h
        
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # aplatissement (batch, 784)
        return self.network(x)
