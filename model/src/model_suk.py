import torch
import torch.nn as nn
import math
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # Reshape the tensor
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # Transpose the tensor
        x = x.transpose(1, 2).contiguous()

        # Reshape back to the original shape
        x = x.view(batch_size, num_channels, height, width)

        return x
class IRShuffleUnit_d(nn.Module):
    def __init__(self, main_channels, side_channels):
        super(IRShuffleUnit_d, self).__init__()
        self.main = main_channels
        self.side = side_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.main[0],self.main[1],kernel_size=1),
            nn.BatchNorm2d(self.main[1]),
            nn.PReLU(),
        )
        self.DWconv_2 = nn.Sequential(
            nn.Conv2d(self.main[1],self.main[2],kernel_size=3,stride=2,groups=self.main[1],padding=1),
            nn.BatchNorm2d(self.main[2]),
            nn.PReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.main[2],self.main[3],kernel_size=1),
            nn.BatchNorm2d(self.main[3]),
        )
        self.DWconv_4 = nn.Sequential(
            nn.Conv2d(self.side[0],self.side[1],kernel_size=3,stride=2,groups=self.side[0],padding=1),
            nn.BatchNorm2d(self.side[1]),
            nn.PReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(self.side[1],self.side[2],kernel_size=1),
            nn.BatchNorm2d(self.side[2]),
        )
        self.channel_shuffle = ChannelShuffle(groups=3)
    def forward(self, x):
        # Main branch
        x_m = x
        x_s = x
        x_add = x_m
        x_m = self.conv_1(x_m)
        x_m = self.DWconv_2(x_m)
        x_m = self.conv_3(x_m)
        # if x_m.size(1) == x_add.size(1):
        #     x_m = x_m+x_add[:,:,::2,::2] + x_add[:,:,1::2,1::2]
        # else:
        #     x_m = x_m+x_add[:,::2,::2,::2] +x_add[:,1::2,::2,::2]+ x_add[:,::2,1::2,1::2] + x_add[:,1::2,1::2,1::2]
        x_s = self.DWconv_4(x_s)
        x_s = self.conv_5(x_s)

        out = torch.cat((x_m,x_s),dim = 1)

        out = self.channel_shuffle(out)
        return out

class IRShuffleUnit_c(nn.Module):
    def __init__(self, main_channels):
        super(IRShuffleUnit_c, self).__init__()
        self.main = main_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.main[0],self.main[1],kernel_size=1),
            nn.BatchNorm2d(self.main[1]),
            nn.PReLU(),
        )
        self.DWconv_2 = nn.Sequential(
            nn.Conv2d(self.main[1],self.main[2],kernel_size=3,groups=self.main[1],padding=1),
            nn.BatchNorm2d(self.main[2]),
            nn.PReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.main[2],self.main[3],kernel_size=1),
            nn.BatchNorm2d(self.main[3]),
        )
        self.channel_shuffle = ChannelShuffle(groups=3)
    def forward(self, x):
        num_channels = x.size(1)
        x_s = x[:,num_channels//2:,:,:]
        x_m = x[:,:num_channels//2,:,:]
        x_add = x_m
        x_m = self.conv_1(x_m)
        x_m = self.DWconv_2(x_m)
        x_m = self.conv_3(x_m)
        x_m = x_m+x_add
        out = torch.cat((x_m,x_s),dim = 1)

        out = self.channel_shuffle(out)
        return out

class Shuffle(nn.Module):
    def __init__(self):
        super(Shuffle,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.PWConv1 = nn.Conv2d(192,512,kernel_size=1)
        self.DWConv1 = nn.Conv2d(512,512,kernel_size=(7,6),groups=512)
        self.PWConv2 = nn.Conv2d(512,128,kernel_size=1)
        self.ir_shuffle1 = IRShuffleUnit_d([32,96,96,16],[32,64,32])
        self.ir_shuffle2 = IRShuffleUnit_c([24,48,48,24])
        self.ir_shuffle3 = IRShuffleUnit_d([48,192,192,48],[48,96,48]) 
        self.ir_shuffle4 = IRShuffleUnit_c([48,96,96,48])
        self.ir_shuffle5 = IRShuffleUnit_c([48,96,96,48])
        self.ir_shuffle6 = IRShuffleUnit_c([48,96,96,48])
        self.ir_shuffle7 = IRShuffleUnit_d([96,256,256,96],[96,192,96])
        self.ir_shuffle8 = IRShuffleUnit_c([96,192,192,96])
    def forward(self,image):
        out = self.conv1(image)
        out = self.ir_shuffle1(out)
        out = self.ir_shuffle2(out)
        out = self.ir_shuffle3(out)
        out = self.ir_shuffle4(out)
        out = self.ir_shuffle5(out)
        out = self.ir_shuffle6(out)
        out = self.ir_shuffle7(out)
        out = self.ir_shuffle8(out)

        out = self.PWConv1(out)
        out = self.DWConv1(out)
        out = self.PWConv2(out)
        out = out.squeeze(dim=-2).squeeze(dim=-1)
        return out

if __name__ == "__main__": 
    # test_cos()
    net = Shuffle()
    net.eval()
    
        

    
    
