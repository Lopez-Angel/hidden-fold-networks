# Copyright 2021 Angel Lopez Garcia-Arias

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        https://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder
from args import args

__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "WideResNet50",
    "ResNet101",
    "HFN_ResNet50_3_4_ubn",
    "HFN_ResNet50_2_3_4_ubn",
    "HFN_WideResNet50_3_4_ubn",
    "HFN_ResNet101_3_4_ubn",
    "HFN_ResNet152_3_4_ubn",
    "HFN_ResNet200_3_4_ubn",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, wider=1, stride=1):
        super(BasicBlock, self).__init__()

        width = planes * wider

        self.conv1 = builder.conv3x3(in_planes, width, stride=stride)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, self.expansion * planes, stride=1)
        self.bn2 = builder.batchnorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, wider=1, stride=1):
        super(Bottleneck, self).__init__()

        width = planes * wider

        self.conv1 = builder.conv1x1(in_planes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class FoldBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBasicBlock, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv3x3(in_planes, width, stride=stride)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, self.expansion * planes, stride=1)
        self.bn2 = builder.batchnorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        for t in range(self.iters):
            x_i = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x += self.shortcut(x_i)
            x = F.relu(x)
        return x


class FoldBottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBottleneck, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv1x1(in_planes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )
    def forward(self, x):
        for t in range(self.iters):
            x_i = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x += self.shortcut(x_i)
            x = F.relu(x)
        return x


class FoldBasicBlockUBN(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBasicBlockUBN, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv3x3(in_planes, width, stride=stride)
        self.conv2 = builder.conv3x3(width, self.expansion * planes, stride=1)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = builder.conv1x1(in_planes, self.expansion * planes, stride=stride)

        for t in range(self.iters):
            setattr(self, f'bnorm1_{t}', nn.BatchNorm2d(width))
            setattr(self, f'bnorm2_{t}', nn.BatchNorm2d(width))
            if self.shortcut:
                setattr(self, f'bnorm3_{t}', nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        for t in range(self.iters):
            x_i = x

            x = self.conv1(x)
            x = getattr(self, f'bnorm1_{t}')(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = getattr(self, f'bnorm2_{t}')(x)

            if self.shortcut:
                y = self.shortcut(x_i)                
                x += getattr(self, f'bnorm3_{t}')(y)
            else:
                x += x_i

            x = F.relu(x)

        return x


class FoldBottleneckUBN(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBottleneckUBN, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv1x1(in_planes, width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.conv3 = builder.conv1x1(width, self.expansion * planes)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = builder.conv1x1(in_planes, self.expansion * planes, stride=stride)

        for t in range(self.iters):
            setattr(self, f'bnorm1_{t}', nn.BatchNorm2d(width))
            setattr(self, f'bnorm2_{t}', nn.BatchNorm2d(width))
            setattr(self, f'bnorm3_{t}', nn.BatchNorm2d(self.expansion * planes))
            if self.shortcut:
                setattr(self, f'bnorm4_{t}', nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        for t in range(self.iters):
            x_i = x

            x = self.conv1(x)
            x = getattr(self, f'bnorm1_{t}')(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = getattr(self, f'bnorm2_{t}')(x)
            x = F.relu(x)

            x = self.conv3(x)
            x = getattr(self, f'bnorm3_{t}')(x)

            if self.shortcut:
                y = self.shortcut(x_i)                
                x += getattr(self, f'bnorm4_{t}')(y)
            else:
                x += x_i
            
            x = F.relu(x)

        return x


class HFN_ResNet(nn.Module):
    def __init__(self, builder, block, hfn_block, is_hfn_stage, num_blocks, wider=1):
        super(HFN_ResNet, self).__init__()
        self.in_planes = 64
        self.builder = builder
        self.block = block
        self.hfn_block = hfn_block
        self.is_hfn_stage = is_hfn_stage
        self.num_blocks = num_blocks

        #PRE-NET
        self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)
        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      

        #CORE      
        self.stage_1 = self._make_stage(layer_n=0, planes=64, stride=1, wider=wider)
        self.stage_2 = self._make_stage(layer_n=1, planes=128, stride=2, wider=wider)
        self.stage_3 = self._make_stage(layer_n=2, planes=256, stride=2, wider=wider)
        self.stage_4 = self._make_stage(layer_n=3, planes=512, stride=2, wider=wider)

        #POST-NET
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if args.last_layer_dense:
            self.fc = nn.Conv2d(512 * block.expansion, args.n_classes, 1)
        else:
            self.fc = builder.conv1x1(512 * block.expansion, args.n_classes)

    def _make_stage(self, layer_n, planes, stride, wider):
        layers = []

        # Projection block
        layers.append(self.block(self.builder, self.in_planes, planes, wider, stride=stride))
        self.in_planes = planes * self.block.expansion

        # Rest of blocks
        next_block = self.hfn_block if self.is_hfn_stage[layer_n] else self.block
        
        if next_block.__name__.startswith('Fold'):
            layers.append(next_block(self.builder, self.in_planes, planes, wider, stride=1, iters=self.num_blocks[layer_n]-1))
        else:
            for _ in range(self.num_blocks[layer_n] - 1):
                layers.append(next_block(self.builder, self.in_planes, planes, wider, stride=1))
        self.in_planes = planes * next_block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)

        out = self.avgpool(out)
        out = self.fc(out)
        return out.flatten(1)



## Vanilla/HNN models (Non-folded)
def ResNet18():
    return HFN_ResNet(get_builder(), BasicBlock, None, [False, False, False, False], [2, 2, 2, 2])

def ResNet34():
    return HFN_ResNet(get_builder(), BasicBlock, None, [False, False, False, False], [3, 4, 6, 3])

def ResNet50():
    return HFN_ResNet(get_builder(), Bottleneck, None, [False, False, False, False], [3, 4, 6, 3])

def WideResNet50():
    return HFN_ResNet(get_builder(), Bottleneck, None, [False, False, False, False], [3, 4, 6, 3], wider=2)

def ResNet101():
    return HFN_ResNet(get_builder(), Bottleneck, None, [False, False, False, False], [3, 4, 23, 3])


## Folded/HFN models
def HFN_ResNet50_3_4_ubn():
    return HFN_ResNet(get_builder(), Bottleneck, FoldBottleneckUBN, [False, False, True, True], [3, 4, 6, 3])

def HFN_ResNet50_2_3_4_ubn():
    return HFN_ResNet(get_builder(), Bottleneck, FoldBottleneckUBN, [False, True, True, True], [3, 4, 6, 3])

def HFN_WideResNet50_3_4_ubn():
    return HFN_ResNet(get_builder(), Bottleneck, FoldBottleneckUBN, [False, False, True, True], [3, 4, 6, 3], wider=2)

def HFN_ResNet101_3_4_ubn():
    return HFN_ResNet(get_builder(), Bottleneck, FoldBottleneckUBN, [False, False, True, True], [3, 4, 23, 3])

def HFN_ResNet152_3_4_ubn():
    return HFN_ResNet(get_builder(), Bottleneck, FoldBottleneckUBN, [False, False, True, True], [3, 8, 36, 3])

def HFN_ResNet200_3_4_ubn():
    return HFN_ResNet(get_builder(), Bottleneck, FoldBottleneckUBN, [False, False, True, True], [3, 24, 36, 3])
