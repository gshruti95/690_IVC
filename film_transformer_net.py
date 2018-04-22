import torch
from torch.nn.parameter import Parameter
# from torch.autograd import Variable
import enums

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.cin1 = CIN(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.cin2 = CIN(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True) # 4,128,64,64
        self.cin3 = CIN(128)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.cin4 = CIN(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.cin5 = CIN(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X, s_idx=None):
        y = self.relu(self.cin1(self.in1(self.conv1(X)), s_idx))
        y = self.relu(self.cin2(self.in2(self.conv2(y)), s_idx))
        y = self.relu(self.cin3(self.in3(self.conv3(y)), s_idx))
        y = self.res1(y, s_idx)
        y = self.res2(y, s_idx)
        y = self.res3(y, s_idx)
        y = self.res4(y, s_idx)
        y = self.res5(y, s_idx)
        y = self.relu(self.cin4(self.in4(self.deconv1(y)), s_idx))
        y = self.relu(self.cin5(self.in5(self.deconv2(y)), s_idx))
        y = self.deconv3(y)
        return y


class CIN(nn.Module):
    def __init__(self, channels):
        super(CIN, self).__init__()
        self.gammas = nn.Parameter(torch.ones(enums.num_styles, channels)) # s,C
        self.betas = nn.Parameter(torch.zeros(enums.num_styles, channels)) # s,C

    def forward(self, X, s_idx=None):
        # Train mode: X is N,C,H,W 
        # Eval mode: s,C,H,W with no s_idx or 1,C,H,W with s_idx
        if s_idx:
            gamma = self.gammas[s_idx].unsqueeze(0).unsqueeze(2).unsqueeze(3) # 1,C,1,1
            beta = self.betas[s_idx].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            out = gamma*X + beta # N,C,H,W
        else:
            gammas = self.gammas.unsqueeze(2).unsqueeze(3) # s,C,1,1
            betas = self.betas.unsqueeze(2).unsqueeze(3)
            out = gammas*X + betas
            
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.cin1 = CIN(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.cin2 = CIN(channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, s_idx):
        residual = x
        out = self.relu(self.cin1(self.in1(self.conv1(x)), s_idx))
        out = self.cin2(self.in2(self.conv2(out)), s_idx)
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            # self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out