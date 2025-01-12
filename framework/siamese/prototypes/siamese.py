import numpy as np
import random
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torchvision import models

from torch.utils.data import Dataset, DataLoader, random_split


classes = [
    'meningioma', # 1
    'notumor', # 0
]

# Custom Subset class to maintain the original dataset attributes
class SiameseSubset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices
        self.transform = original_dataset.transform
        self.n_pairs = original_dataset.n_pairs
        self.pairs = [original_dataset.pairs[i] for i in indices]
        self.pair_labels = [original_dataset.pair_labels[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]

class SiameseBrainDataset(Dataset):
    def __init__(self, path, n_pairs=1000, transform=None):
        self.path = path
        self.transform = transform
        self.n_pairs = n_pairs
        
        # Store images paths and labels
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for i, cls in enumerate(classes):
            for file in os.listdir(os.path.join(path, cls)):
                self.images.append(os.path.join(path, cls, file))
                self.labels.append(i)
                
        # Convert to numpy arrays for easier indexing
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        # Create pairs
        self.create_pairs()
        
    def create_pairs(self):
        self.pairs = []
        self.pair_labels = []
        
        # Generate similar pairs (same class)
        n_same = self.n_pairs // 2
        for _ in range(n_same):
            # Pick a random class
            c = random.choice(np.unique(self.labels))
            # Find all indices for this class
            idx = np.where(self.labels == c)[0]
            # Pick two random images from this class
            if len(idx) >= 2:
                i1, i2 = random.sample(list(idx), 2)
                self.pairs.append((self.images[i1], self.images[i2]))
                self.pair_labels.append(1)  # Similar pair
        
        # Generate dissimilar pairs (different classes)
        for _ in range(self.n_pairs - n_same):
            # Pick two different classes
            c1, c2 = random.sample(list(np.unique(self.labels)), 2)
            # Find indices for each class
            idx1 = np.where(self.labels == c1)[0]
            idx2 = np.where(self.labels == c2)[0]
            # Pick one random image from each class
            i1 = random.choice(idx1)
            i2 = random.choice(idx2)
            self.pairs.append((self.images[i1], self.images[i2]))
            self.pair_labels.append(0)  # Dissimilar pair
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.pair_labels[idx]
        
        # Load images
        img1 = plt.imread(img1_path)
        img2 = plt.imread(img2_path)
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Stack images along a new dimension
        image_pair = torch.stack([img1, img2])
        
        return image_pair, label

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        # Load two pretrained ResNet50 architectures
        self.resnet = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer with an embedding layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, embedding_dim)
        # self.resnet2.fc = nn.Linear(num_features, embedding_dim)
        
        # Add a normalization layer for embeddings
        self.l2_norm = nn.functional.normalize

    def forward_once(self, x, branch="resnet1"):
        """Forward pass through one of the ResNet branches."""
        if branch == "resnet1":
            return self.l2_norm(self.resnet(x), dim=1)
        elif branch == "resnet2":
            return self.l2_norm(self.resnet(x), dim=1)
        else:
            raise ValueError("branch should be 'resnet1' or 'resnet2'.")

    def forward(self, x1, x2):
        """Forward pass through the Siamese network."""
        # Pass the inputs through their respective ResNet branches
        embedding1 = self.forward_once(x1, branch="resnet1")
        embedding2 = self.forward_once(x2, branch="resnet2")
        
        return embedding1, embedding2
    
class SiameseNetwork3D(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork3D, self).__init__()
        
        # Load two pretrained ResNet50 architectures
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 113 channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=113,  # Change from 3 to 113
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize the new conv1 weights properly
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Replace the final fully connected layer with an embedding layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, embedding_dim)
        # self.resnet2.fc = nn.Linear(num_features, embedding_dim)
        
        # Add a normalization layer for embeddings
        self.l2_norm = nn.functional.normalize

    def forward_once(self, x, branch="resnet1"):
        """Forward pass through one of the ResNet branches."""
        if branch == "resnet1":
            return self.l2_norm(self.resnet(x), dim=1)
        elif branch == "resnet2":
            return self.l2_norm(self.resnet(x), dim=1)
        else:
            raise ValueError("branch should be 'resnet1' or 'resnet2'.")

    def forward(self, x1, x2):
        """Forward pass through the Siamese network."""
        # Pass the inputs through their respective ResNet branches
        embedding1 = self.forward_once(x1, branch="resnet1")
        embedding2 = self.forward_once(x2, branch="resnet2")
        
        return embedding1, embedding2