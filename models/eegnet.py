'''
Pytorch implementation of EEGNet from the official githup repo from the paper:
- https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
- https://github.com/vlawhern/arl-eegmodels / https://arxiv.org/pdf/1611.08024

'''

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


class KerasBasicDropuout2D(nn.Module):
    def __init__(self, p):
        super().__init__()
        
        # Keras’ SpatialDropout1D in PyTorch:
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = F.dropout2d(x, p, training=self.training)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]


    def forward(self):
        pass

class ConstrainedConv2d(nn.Conv2d):
    """L2Norm constrained Conv2D block"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            input = x,
            weight = self.weight.clamp(max=1, min = -1),
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups,
            )
        return x


class DepthwiseConv2D(nn.Module):
    def __init__(self, config: dataclass):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels= config.F1,
            out_channels = config.F2,#config.D,
            kernel_size = (config.channels, 1),
            groups =config.F1,
            stride = 1,
            padding = 0,#'same',
            bias = config.bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        print(self.depth_conv.weight)
        return x


class SeparableConv2D(nn.Module):
    def __init__(self, config: dataclass):
        super().__init__()
        self.sepconv = nn.Conv2d(
                in_channels = config.F2,
                out_channels = config.F2,
                kernel_size = 1,
                stride = 1,
                padding = 'same',
                bias = config.bias,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sepconv(x)
        return x
    

class Conv2D(nn.Module):
    def __init__(self, config: dataclass):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = 1,
            out_channels = config.F1,
            kernel_size = (1, config.kernLength),
            stride = 1,
            padding = 'same',
            bias = config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EEGNet(nn.Module):
    def __init__(self, config: dataclass):
        super().__init__()
        self.config = config
        self.F2 = config.F1 * config.D

        if config.dropoutType == 'spatial':
            # Pytoch dropout2d == keras spatialdropout2d
            dropoutType = nn.Dropout2d(config.dropoutRate) 
        
        elif config.dropoutType == 'base':
            # TODO keras implementation of basic dropout2d is different than Pytorchs!
            dropoutType = nn.Dropout(config.dropoutRate)
        
        else:
            raise ValueError("dropoutType must be 'spatial' or 'base'.")
        
        # block 1
        self.block1 = nn.Sequential(
            Conv2D(config),
            nn.BatchNorm2d(config.F1),
            DepthwiseConv2D(config),
            nn.BatchNorm2d(config.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutType,
        )
        
        # block 2
        self.block2 = nn.Sequential(
            SeparableConv2D(config),
            nn.BatchNorm2d(config.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropoutType,
        )
        
        # classification block
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
            in_features = self._get_num_inputs(),
            out_features = config.n_classes,
            ),
            nn.Softmax(),
        )

    def _get_num_inputs(self):
        """Mocks input shape for last linear layer dynamically."""
        with torch.no_grad():
            dummy_data = torch.zeros(1, 1, self.config.channels, self.config.samples)

            dummy_data = self.block1(dummy_data)
            dummy_data = self.block2(dummy_data)

        return self.F2 * dummy_data.shape[3]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shape of input data should be (Batch, 1, Channels, Samples)."""

        # block 1
        x = self.block1(x)

        # block 2
        x = self.block2(x)

        # classification
        x = self.classify(x)
        return x