import torch.nn.functional as F
from torch import nn 

# simple CNN model
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 16):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.4)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2304, 256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self,x):
        x = x.view(x.size(0), 1, 64, 151) #reshape the input tensor
        x = self.dropout1(self.batch_norm1(F.relu(self.conv1(x))))
        x = self.max_pool1(x)
        x = self.dropout2(self.batch_norm2(F.relu(self.conv2(x))))
        x = self.max_pool2(x)
        x = self.dropout3(self.batch_norm3(F.relu(self.conv3(x))))
        x = self.max_pool3(x)
        x = self.flatten(x) 
        x = self.dropout4(self.batch_norm4(F.relu(self.fc1(x))))
        x = self.fc2(x)
        return x 