import torch
import torch.nn as nn
import math
import numpy as np
import cv2 as cv
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import transforms

# http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.PReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.PReLU(),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleFaceNet(nn.Module):
    def __init__(self, stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024], inverted_residual=InvertedResidual):
        super(ShuffleFaceNet, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(),
        )
        input_channels = output_channels

        self.gdc = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=1, padding=0, bias=False, groups=input_channels),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(),
        )

        input_channels = output_channels
        output_channels = 128

        self.linearconv = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(output_channels)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = nn.functional.interpolate(x, size=[112, 112])
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        #x = x.mean([2, 3])  # globalpool 
        x = self.gdc(x)
        # x = np.squeeze(x, axis=2)
        x = x.view(x.size(0),1024, 1)
        x = self.linearconv(x)
        x = x.view(x.size(0), 128, 1, 1)
        x = self.bn(x)
        x = x.view(x.size(0), -1)


        return x

    def forward(self, x):
        return self._forward_impl(x)

def test_cos():
    trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    def load_img(img_path):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = trans(img).unsqueeze(0)
        return img
    
    device = 'cuda:1'
    input1 = load_img('/home/ipcteam/congnt/face/face_recognition/data/data_umd/img/8264/331161.jpg').to(device)
    input2 = load_img('/home/ipcteam/congnt/face/face_recognition/data/data_umd/img/8266/39568.jpg').to(device)
    input3 = load_img('/home/ipcteam/congnt/face/face_recognition/data/data_umd/img/8264/331169.jpg').to(device)
    net = ShuffleFaceNet()
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load('/home/ipcteam/congnt/face/face_recognition/model/weights/Shuffle/060.ckpt')['net_state_dict'])
    with torch.no_grad():
        output1 = net(input1)
        output2 = net(input2)
        output3 = net(input3)
        cosin_neg = F.cosine_similarity(output1,output2)
        cosin_pos = F.cosine_similarity(output1,output3)
        print("Negative cos:",cosin_neg.item())
        print("Positive cos:",cosin_pos.item())
def convert_onnx():
    net = ShuffleFaceNet()
    net.eval()
    input = torch.randn(1,3,112,112)
    net.load_state_dict(torch.load('/home/ipcteam/congnt/face/face_recognition/model/weights/Shuffle/Vn/190.ckpt')['net_state_dict'])
    torch.onnx.export(net,
                      input,
                      'custom.onnx',
                      opset_version=11,    
                      input_names=['input'],
                      output_names=['output'])
if __name__ == "__main__":
    convert_onnx()