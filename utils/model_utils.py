import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input channels = 1 (grayscale image)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after two max-pooling layers, the size is (28/2/2=7)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for the MNIST dataset

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling with 2x2 window
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Use log softmax for better numerical stability

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input channels = 1 (grayscale image)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after two max-pooling layers, the size is (28/2/2=7)
        self.bn_fc1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 10)  # 10 classes for the MNIST dataset

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Max pooling with 2x2 window
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Use log softmax for better numerical stability

class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layer
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 128 channels, 8x8 image dimensions after 3 layers & pooling
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply first convolutional layer, then batch norm, then activation, then max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Apply second convolutional layer, then batch norm, then activation, then max pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Apply third convolutional layer, then batch norm, then activation
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.fc2(x)
        return x