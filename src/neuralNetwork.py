import torch
from torch import nn
class GroupNet(nn.Module):
    def __init__(self,input_size,hidden1_size,num_classes):
        super(GroupNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1,1)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)

        return