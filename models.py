import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_channel,block_channel,stride=1,downsample = None):
        super(BasicBlock,self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel,block_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_channel,block_channel * self.expansion,kernel_size=3,stride = stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(block_channel)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu2(out)
        return out
    
class Resnet(nn.Module):
    def __init__(self,in_channel=1,num_classes=2,block=BasicBlock,num_blocks = [2,2,2,2]):
        super(Resnet,self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channel,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block, 64 ,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block, 128 ,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.global_avg_pool =  nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion,num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)

        x = torch.flatten(x,1)

        x = self.fc(x)

        return x
    
    def _make_layer(self,block,block_channel,block_num,stride):
        layers = []
        downsample = nn.Conv2d(self.in_channel,block_channel * block.expansion,kernel_size=1,stride=stride,bias=False)

        layers.append(block(self.in_channel,block_channel,stride,downsample))
        self.in_channel = block_channel * block.expansion
    
        for _ in range(1,block_num):
          layers.append(block(self.in_channel,block_channel,stride=1))
          return nn.Sequential(*layers)

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels = 256):
        super(FPN,self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01)
        ) for in_ch in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Dropout(0.01)
        ) for _ in in_channels_list])
        self.small_target_branch = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear",align_corners = True)

        for m in self.lateral_convs + self.fpn_convs + [self.small_target_branch]:
            for sub_m in m:
                if isinstance(sub_m,nn.Conv2d):
                    nn.init.kaiming_normal_(sub_m.weight,mode='fan_out',nonlinearity='relu')
                    if sub_m.bias is not None:
                        nn.init.constant_(sub_m.bias,0)
                elif isinstance(sub_m,nn.BatchNorm2d):
                    nn.init.constant_(sub_m.weight,1)
                    nn.init.constant_(sub_m.bias,0)

    def forward(self,x_list):
        lateral_feats = [lateral_conv(x) for lateral_conv,x in zip(self.lateral_convs,x_list)]
        fpn_feats = []
        prev_feat = lateral_feats[-1]
        fpn_feats.append(self.fpn_convs[-1](prev_feat))
        for i in range(len(lateral_feats)-2,-1,-1):
            prev_feat = self.upsample(prev_feat)
            if prev_feat.shape[2:] != lateral_feats[i].shape[2:]:
                prev_feat = F.interpolate(prev_feat, size=lateral_feats[i].shape[2:],mode='bilinear',align_corner=True)
            prev_feat = prev_feat + lateral_feats[i]
            fpn_feats.insert(0,self.fpn_convs[i](prev_feat))

        fpn_feats[0] = self.small_target_branch(fpn_feats[0])
        return fpn_feats
    
    