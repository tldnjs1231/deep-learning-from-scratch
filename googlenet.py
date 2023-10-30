import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000, init_weights=True):
        super(GoogLeNet, self).__init__()

        # aux_logits must be set
        # assert (condition), (message to raise otherwise; optional)
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        
        # Regular convolutions
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception modules (9 blocks): 3 + aux1 + 3 + aux2 + 3
        # Parameters: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool
        self.inceptionA = nn.Sequential(
            Inception_block(192, 64, 96, 128, 16, 32, 32),
            Inception_block(256, 128, 128, 192, 32, 96, 64)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inceptionB1 = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inceptionB2 = nn.Sequential(
            Inception_block(512, 160, 112, 224, 24, 64, 64),
            Inception_block(512, 128, 128, 256, 24, 64, 64),
            Inception_block(512, 112, 144, 288, 32, 64, 64)
        )
        self.inceptionB3 = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inceptionC = nn.Sequential(
            Inception_block(832, 256, 160, 320, 32, 128, 128),
            Inception_block(832, 384, 192, 384, 48, 128, 128)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        # Final classifier
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        # Auxiliary classifiers
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes) # after inceptionB1
            self.aux2 = AuxiliaryClassifier(528, num_classes) # after inceptionB2
        else:
            self.aux1 = self.aux2 = None
        
        if init_weights:
            self._init_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inceptionA(x) # first, second inception block
        x = self.maxpool3(x)
        x = self.inceptionB1(x) # third inception block

        if self.aux_logits and self.training: # whether this module is in train or eval mode (boolean)
            aux1 = self.aux1(x)
        
        x = self.inceptionB2(x) # fourth, fifth, sixth inception block

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inceptionB3(x) # seventh inception block
        x = self.maxpool4(x)
        x = self.inceptionC(x) # eighth, ninth inception block

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        out = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, out
        else:
            return out
        
    def _init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1_pool, kernel_size=1)
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)

        return out


if __name__ == '__main__':
    batch_size = 5
    x = torch.randn(batch_size, 3, 224, 224)
    
    model = GoogLeNet(aux_logits=True, num_classes=1000)

    aux1 = model(x)[0].shape
    aux2 = model(x)[1].shape
    out = model(x)[2].shape
    
    assert out == torch.Size([batch_size, 1000])
    print(aux1, aux2, out)

