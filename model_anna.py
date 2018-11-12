import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ''' First Block '''
        self.conv2d_1   = nn.Conv2d(3, 10, (3,3))
        self.conv2d_2   = nn.Conv2d(10,20, (3,3))
        self.maxpool    = nn.MaxPool2d((2,2))
        self.dropout    = nn.Dropout(0.25)

        ''' Second Block '''
        self.conv2d_3   = nn.Conv2d(20,40, (3,3))
        self.conv2d_4   = nn.Conv2d(40,80, (3,3))

        ''' Third Block '''
        self.conv2d_5   = nn.Conv2d(80,30, (3,3))
        self.conv2d_6   = nn.Conv2d(30,15, (3,3))

        ''' Linear Layer '''
        self.fc1        = nn.Linear(6000, 1000)
        self.drouput2   = nn.Dropout(0.4)
        self.fc2        = nn.Linear(1000, 300)
        self.fc3        = nn.Linear(300, 26+3) #letters in alphabet + space + point


    def forward(self, x):

        ''' First conv block '''
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_1(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        ''' Second conv block '''
        x = F.relu(self.conv2d_3(x))
        x = F.relu(self.conv2d_4(x))
        x = self.maxpool(x)
        x = self.maxpool(x)

        ''' Third conv block '''
        x = F.relu(self.conv2d_5(x))
        x = F.relu(self.conv2d_6(x))
        x = self.maxpool(x)
        x = self.maxpool(x)

        ''' Flatten and Fully Connect'''
        x = x.reshape(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.drouput2(x)
        x = self.fc3(x)
        return x
