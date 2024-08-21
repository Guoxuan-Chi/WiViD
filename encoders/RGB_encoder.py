import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.silu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return self.silu(x + output)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResNetDownBlock, self).__init__()
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        extra_x = self.extra(x)
        out = self.conv1(x)
        out = self.silu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        return self.silu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=[7,7], stride=[3,2], padding=[3,3])
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, [3, 3], [1, 1], [1, 1]),
                                    ResNetBasicBlock(64, 64, [3, 3], [1, 1], [1, 1]))
        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, [1, 3, 3], [2, 2, 1], [0, 1, 1]),
                                    ResNetBasicBlock(128, 128, [3, 3], [1, 1], [1, 1]))
        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [1, 3, 3], [2, 2, 1], [0, 1, 1]),
                                    ResNetBasicBlock(256, 256, [3, 3], [1, 1], [1, 1]))
        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, [1, 3, 3], [2, 2, 1], [0, 1, 1]),
                                    ResNetBasicBlock(512, 512, [3, 3], [1, 1], [1, 1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = rearrange(x, "b d 1 1 -> b d")
        return x


class RGB_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet18()

    def forward(self, x):

        x = self.resnet(x)
        return x
    

# if __name__ == "__main__":

#     device = "cuda:0"
#     model = RGB_encoder().to(device)
#     para = sum(p.numel() for p in model.parameters())
#     print("Num params: ", para) 
#     print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

#     # print(model)
#     img = torch.randn((64, 3, 320, 180), device=device)

#     print(img.shape)
#     output= model(img)
#     print(output.shape)