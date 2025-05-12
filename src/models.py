import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = Conv(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvBlockR(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(ConvBlockR, self).__init__()
        self.conv1 = Conv(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = Conv(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, num_filters)
        self.pool       = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return p, x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(DecoderBlock, self).__init__()
        self.upconv    = nn.ConvTranspose2d(in_channels, num_filters, kernel_size=2, stride=2, padding=0)
        self.bn        = nn.BatchNorm2d(num_filters)
        self.relu      = nn.ReLU(inplace=True)
        self.conv_block = ConvBlockR(num_filters *2, num_filters) ##

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x
      
class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(UNet, self).__init__()
        # Encoder blocks
        self.enc1  = EncoderBlock(in_channels, 64) # 1, 128, 128, 128
        self.enc2  = EncoderBlock(64, 128)
        self.enc3  = EncoderBlock(128, 256)
        self.enc4  = EncoderBlock(256, 512)
        
        self.center = ConvBlock(512, 1024)
        
        # Decoder blocks
        self.dec4  = DecoderBlock(1024, 512) # [1, 512, 32, 32]
        self.dec3  = DecoderBlock(512, 256)
        self.dec2  = DecoderBlock(256, 128)
        self.dec1  = DecoderBlock(128, 64)
        # Output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p1, c1 = self.enc1(x)
        p2, c2 = self.enc2(p1)
        p3, c3 = self.enc3(p2) # [1, 256, 32, 32]
        p4, c4 = self.enc4(p3) # [1, 512, 16, 16], [1, 512, 32, 32]

        center = self.center(p4)
        
        d4 = self.dec4(center, c4)
        d3 = self.dec3(d4, c3)
        d2 = self.dec2(d3, c2)
        d1 = self.dec1(d2, c1)

        out = self.final(d1)
        return self.sigmoid(out)
