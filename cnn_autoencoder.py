"""
This class module uses the torch.nn.Module to create two components of the autoencoder.
The architecture consist convolutional and max pooling layers to get the features compressed to the encoding dimension.
Between the hidden layers, the ReLu activation is used. The final activation is a sigmoid function.
The architecture is based on the code: 
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Convolutional_Autoencoder_Solution.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2= nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        self.d_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.d_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        
    def forward(self, x):
        x = F.relu(self.d_conv1(x))
        x = torch.sigmoid(self.d_conv2(x))
        return x
