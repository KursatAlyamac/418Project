import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.in_channels == self.out_channels:
          inputx = x
          x = self.conv1(x)
          x = F.leaky_relu(x)
          x = self.conv2(x)
          x = inputx + x
          x = F.leaky_relu(x)
        else:
          x = self.conv1(x)
          conv1x = x
          x = F.leaky_relu(x)
          x = self.conv2(x)
          x = conv1x + x
          x = F.leaky_relu(x)
          
        return x

class Actor (nn.Module):
    def __init__(self, block):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(64)

        self.linear = nn.Linear(64 + 2, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, in_planes, out_planes):
        layers = [
            block(in_planes, out_planes),
            block(out_planes, out_planes)
        ]
        return nn.Sequential(*layers)

    def forward(self, x, aim_point):
        #@img: (B,3,96,128)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x)

        x = F.avg_pool2d(x,4) # (B, 16, 48, 64)
        x = F.avg_pool2d(x, kernel_size=(24, 32)) #(B, 16, 3, 3)

        x = torch.cat((torch.flatten(x,1), aim_point), dim=1)

        x = self.linear(x)

        return self.sigmoid(x)

class Critic (nn.Module):
    def __init__(self, block):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, 64)

        self.layer2 = self._make_layer(block, 64, 64)

        self.layer3 = self._make_layer(block, 64, 16)
        self.layer4 = self._make_layer(block, 16, 16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.linear = nn.Linear(16 + 2 + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, in_planes, out_planes):
        layers = [
            block(in_planes, out_planes),
            block(out_planes, out_planes)
        ]
        return nn.Sequential(*layers)

    def forward(self, image, aim_point, action):

        #@img: (B,3,96,128)
        x = image
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = F.leaky_relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)


        x = F.avg_pool2d(x,2) # (B, 16, 48, 64)
        x = self.layer3(x)
        x = F.avg_pool2d(x,2) # (B, 16, 24, 32)
        x = self.layer4(x)
        x = F.avg_pool2d(x,(24, 32)) #(B, 16, 1, 1)
        x = self.conv2(x)

        x = torch.cat((x.squeeze(), aim_point, action), dim=1)
        x = self.linear(x.squeeze())


        return x
