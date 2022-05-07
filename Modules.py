import torch.nn as nn
from collections import namedtuple
import random

simple_Transition = namedtuple(
    "simple_Transition", ("state", "action", "reward"))


class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0
        random.seed(20)

    def push(self, *args):
        if len(self.memory) < self.size:
            self.memory.append(None)

        self.memory[self.position] = simple_Transition(*args)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        rand_samples = random.sample(self.memory, batch_size - 1)
        rand_samples.append(self.memory[self.position - 1])
        return rand_samples

    def get(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        if inplanes != planes:
            self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv3:
            out += self.conv3(x)
        else:
            out += x

        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Perception_Module(nn.Module):
    def __init__(self):
        super(Perception_Module, self).__init__()
        self.C1 = conv3x3(4, 64)
        self.MP1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.RB1 = BasicBlock(64, 128)
        self.MP2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.RB2 = BasicBlock(128, 256)
        self.RB3 = BasicBlock(256, 512)

    def forward(self, x):
        x = self.C1(x)
        x = self.MP1(x)
        x = self.RB1(x)
        x = self.MP2(x)
        x = self.RB2(x)
        x = self.RB3(x)

        return x


class Grasping_Module(nn.Module):
    def __init__(self, act_dim_2=6):
        super(Grasping_Module, self).__init__()
        self.RB1 = BasicBlock(512, 256)
        self.RB2 = BasicBlock(256, 128)
        self.UP1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.RB3 = BasicBlock(128, 64)
        self.UP2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.C1 = nn.Conv2d(64, act_dim_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.RB1(x)
        x = self.RB2(x)
        x = self.UP1(x)
        x = self.RB3(x)
        x = self.UP2(x)
        x = self.C1(x)
        x.squeeze_()

        return self.sigmoid(x)


def MULTIDISCRETE_RESNET(number_actions_dim_2):
    return nn.Sequential(
        Perception_Module(), Grasping_Module(act_dim_2=number_actions_dim_2)
    )
