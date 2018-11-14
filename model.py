import torch.nn as nn
from torch import functional

class ASLCNN(nn.Module):
    def __init__(self):
        super(ASLCNN, self).__init__()

        self.output_size = 29

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, (5, 5), stride=2),
            nn.ReLU(),
            nn.Conv2d(10, 15, (10, 10), stride=2),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.25)
        )

        self.linear = nn.Sequential(
            nn.Linear(7260, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

