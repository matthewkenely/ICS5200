import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from random import randint

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        
        # Load the pre-trained ResNet50 model
        self.resnet = models.resnet101(pretrained=True)

        # Modify the ResNet model to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last layer

        # Freeze all layers of ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers for fine-tuning
        for layer in list(self.resnet.children())[:]:  # Get the last two layers
            for param in layer.parameters():  # Access parameters of the layer
                param.requires_grad = True

        # Add additional layers: 2 fully connected layers and an output layer
        self.fc1 = nn.Linear(2048, 512)  # 2048 is the output of the last ResNet layer
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization for the first fully connected layer
        self.fc2 = nn.Linear(512, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)  # Batch normalization for the output layer

        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output from ResNet

        # Forward pass through the additional fully connected layers with batch normalization
        x = self.dropout(self.fc1(x))
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.fc2(x)
        x = self.bn2(x)

        # Apply LogSoftmax to get log probabilities
        x = self.softmax(x)

        return x