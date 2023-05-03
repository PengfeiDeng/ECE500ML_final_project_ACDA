import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torch.nn.parameter import Parameter
import math
import scipy as sp
import scipy.linalg as linalg
import numpy as np
import pdb
from torch.nn.utils import spectral_norm

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ACDA, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Filter-generating network
        self.filter_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * kernel_size * kernel_size, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Generate dynamic filters for each pixel
        filters = self.filter_gen(x)
        filters = filters.view(batch_size, self.out_channels, self.kernel_size * self.kernel_size, height, width)
        
        # Perform adaptive convolution
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)
        
        x_unfold = x_unfold.view(batch_size, self.out_channels, self.kernel_size * self.kernel_size, height, width)
        
        out = (filters * x_unfold).sum(dim=2)
        out = out.view(batch_size, self.out_channels, height, width)

        return out


