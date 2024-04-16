"""
This class module uses the torch.nn.Module to create two components of the autoencoder.
The architecture consist of two hidden layers with 500 and 256 neurons, activated by ReLu.
The final activation is a sigmoid function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()       
        self.flatten = nn.Flatten(start_dim=1)
        self.l1 = nn.Linear(784, 500)
        self.l2 = nn.Linear(500, 256)
        self.l3 = nn.Linear(256, encoding_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        # linear section
        self.l1 = nn.Linear(encoding_dim, 256)
        self.l2 = nn.Linear(256, 500)
        self.l3 = nn.Linear(500, 784)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1,28,28))
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = torch.sigmoid(self.unflatten(x))
        return x