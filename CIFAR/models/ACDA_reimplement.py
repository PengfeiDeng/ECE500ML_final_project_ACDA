import torch.nn as nn
import torch.nn.functional as F
from ACDA.Conv_modified import *

class ACDA_modified(nn.Module):
    def __init__(self,out_size):
        super(ACDA_modified, self).__init__()
        self.conv1 = Conv_DCFD(3, 6, kernel_size=5)
        self.conv2 = Conv_DCFD(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(1936, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

