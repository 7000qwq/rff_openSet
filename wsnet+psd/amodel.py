import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, classes):
        super(CNN, self).__init__()
        # First layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, ceil_mode=True)
        # Second layer
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, ceil_mode=True)
        # Third layer
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0, ceil_mode=True)
        self.linear1 = nn.Linear(512, classes)
        self.dropout = nn.Dropout(0.1)
        self.logit = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  #flatten
        x = self.linear1(x)
        x = self.logit(x)
        return x