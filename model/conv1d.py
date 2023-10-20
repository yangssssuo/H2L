import torch.nn as nn
import torch
import numpy as np



class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=10,kernel_size=3,stride=2)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv1d(10,20,3,2)
        self.max_pool2 = nn.MaxPool1d(3,2)
        self.conv3 = nn.Conv1d(20,40,3,2)

        self.linear1 = nn.Linear(4960, 2480)
        self.linear2 = nn.Linear(2480,2480)
        self.linear3 = nn.Linear(2480,1000)

    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.dropout(x,p=0.3)
        x = self.max_pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.dropout(x,p=0.3)
        x = self.max_pool2(x)
        x = nn.functional.relu(self.conv3(x))

        x = x.view(-1,4960)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x