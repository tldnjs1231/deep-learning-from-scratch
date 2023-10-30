import torch
import torch.nn as nn


def conv_1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv_3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            conv_3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            conv_3x3(out_channels, out_channels * self.expansion),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        x = self.residual(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        out = self.relu(x)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            conv_1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            conv_3x3(out_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            conv_1x1(out_channels, out_channels * self.expansion),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        x = self.residual(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        out = self.relu(x)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, img_channels=3, init_weights=True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # output size 1 x 1 for any input size (global average pooling)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.initial(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)

        return out

    def _make_layer(self, block, out_channels, num_blocks, stride):
        identity_downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion: # stride = (1, 2, 2, 2)
            identity_downsample = nn.Sequential(
                conv_1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))

        # Update in_channels
        self.in_channels = out_channels * block.expansion

        for _ in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Kaiming He
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    out = model(x).to(device)

    assert out.shape == torch.Size([batch_size, 1000])
    print(out.shape)

