import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import enums

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        #self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.cin1 = CIN(32)
        #self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.cin2 = CIN(64)
        #self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True) # 4,128,64,64
        self.cin3 = CIN(128)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        #self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.cin4 = CIN(64)
        #self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.cin5 = CIN(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

        # Dilated layers
        self.conv1 = DilConvLayer(3, 32, kernel_size=5, stride=1, dilation=2) # 32,256,256
        self.conv2 = DilConvLayer(32, 64, kernel_size=1, stride=2, dilation=2, ref_pad=False) # 64,128,128 (129 if ref pad)
        self.conv3 = DilConvLayer(64, 128, kernel_size=1, stride=2, dilation=2, ref_pad=False) # 128,64,64 (65 otherwise) 

        # Transpose conv layers
        #self.deconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1) # request output_size=(N,64,128,128)
        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, output_padding=1) # 64,128,128 (actual 127)

        #self.deconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2) # request output_size=(N,64,128,128) (actual 127)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=1, stride=2, output_padding=1) # 32,256,256


    def forward(self, X, s_idx=None):
        y = self.relu(self.cin1(self.in1(self.conv1(X)), s_idx))
        #print y.size()
        y = self.relu(self.cin2(self.in2(self.conv2(y)), s_idx))
        #print y.size()
        y = self.relu(self.cin3(self.in3(self.conv3(y)), s_idx))
        #print y.size()
        y = self.res1(y, s_idx)
        #print y.size()
        y = self.res2(y, s_idx)
        #print y.size()
        y = self.res3(y, s_idx)
        #print y.size()
        y = self.res4(y, s_idx)
        #print y.size()
        y = self.res5(y, s_idx)
        #print y.size()
        y = self.relu(self.cin4(self.in4(self.deconv1(y)), s_idx))
        #print y.size()
        y = self.relu(self.cin5(self.in5(self.deconv2(y)), s_idx))
        #print y.size()
        y = self.deconv3(y)
        #print y.size()
        return y


class CIN(torch.nn.Module):
    def __init__(self, channels):
        super(CIN, self).__init__()
        self.gammas = torch.nn.Parameter(torch.ones(enums.num_styles, channels)) # s,C
        self.betas = torch.nn.Parameter(torch.zeros(enums.num_styles, channels)) # s,C

    def forward(self, X, s_idx, s_list=enums.s_list):
        # Train mode: X is N,C,H,W -- s_idx not None
        # Eval mode: output 1,C,H,W -- s_list not None OR s,C,H,W -- s_list None
        if s_idx is not None:
            gamma = self.gammas[s_idx].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            beta = self.betas[s_idx].unsqueeze(0).unsqueeze(2).unsqueeze(3) # 1,C,1,1
            out = gamma*X + beta # N,C,H,W
        else: # eval mode
            if s_list is not None: # X is 1,C,H,W
                s_wts = Variable(torch.Tensor(s_list), volatile=True, requires_grad=False).unsqueeze(1) # s,1
                if enums.cuda:
                    s_wts = s_wts.cuda()
                gammas = s_wts*self.gammas # s,1*s,C = s*C
                betas = s_wts*self.betas
                if enums.spat is None:
                    gamma = 0.
                    beta = 0.
                    for idx, _ in enumerate(s_wts):
                        gamma += gammas[idx] # C
                        beta += betas[idx]
                    gamma = gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3) # 1,C,1,1
                    beta = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    out = gamma*X + beta
                else: # spatial transfer either V for Vertical or H for horizontal split
                    # count how many styles to split by and chunk X
                    style_ids = [i for i, e in enumerate(s_list) if e!=0]
                    parts = len(style_ids)
                    if enums.spat == 1: # Ver -- chunk by last dim width W
                        split_dim = 3
                    else: # Hor -- chunk by height H
                        split_dim = 2
                    #print X.size(), parts
                    x_list = X.chunk(parts, split_dim)
                    seq = []
                    for i, item in enumerate(x_list):
                        output = gammas[style_ids[i]].unsqueeze(0).unsqueeze(2).unsqueeze(3)*item + betas[style_ids[i]].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                        seq.append(output)
                    out = torch.cat(seq, split_dim)
            else: # X is s,C,H,W
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


class DilConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, ref_pad=True):
        super(DilConvLayer, self).__init__()
        self.ref_pad = ref_pad
        if ref_pad:
            reflection_padding = kernel_size - 1
            self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        x_in = x
        if self.ref_pad:
            x_in = self.reflection_pad(x)
        out = self.conv2d(x_in)
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
