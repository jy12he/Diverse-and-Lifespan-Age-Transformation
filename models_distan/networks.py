### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools
from torch.autograd import grad as Grad
from torch.autograd import Function
import numpy as np
from math import sqrt
from pdb import set_trace as st
import math

###############################################################################
# Functions
###############################################################################





class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur_swap(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur_swap(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d_swap(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)




def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'pixel':
        norm_layer = PixelNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, n_downsample_global=2,
             id_enc_norm='pixel', gpu_ids=[], padding_type='reflect',
             style_dim=50, init_type='gaussian',
             conv_weight_norm=False, decoder_norm='pixel', activation='lrelu',
             adaptive_blocks=4, normalize_mlp=False, modulated_conv=False):

    id_enc_norm = get_norm_layer(norm_type=id_enc_norm)

    netG = Generator(input_nc, output_nc, ngf, n_downsampling=n_downsample_global,
                     id_enc_norm=id_enc_norm, padding_type=padding_type, style_dim=style_dim,
                     conv_weight_norm=conv_weight_norm, decoder_norm=decoder_norm,
                     actvn=activation, adaptive_blocks=adaptive_blocks,
                     normalize_mlp=normalize_mlp, modulated_conv=modulated_conv)

    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])

    netG.apply(weights_init(init_type))

    return netG
# 1
def define_landmarks_G():
    netLandG = TALLSIREN(input_dim=32, z_dim=64, hidden_dim=256, output_dim=32, device=None)
    print('netLandG:')
    print(netLandG)
    # 1
    netLandG.cuda(0)

    return netLandG

# 1  定义 landmarks 的  style encoder
def define_StyleEncoder_L():
    netStyleEncoder_L = StyleEncoder_L()
    print('netStyleEncoder_L:')
    print(netStyleEncoder_L)
    netStyleEncoder_L.cuda(0)

    return netStyleEncoder_L

#  定义 landmarks 的 mapping net
def define_MappingNet_L():
    netMapping_L = MappingNetwork_L()
    print('netMapping_L:')
    print(netMapping_L)
    netMapping_L.cuda(0)

    return netMapping_L



def define_StyleEncoder():
    netStyleEncoder = StyleEncoder()
    print('netStyleEncoder:')
    print(netStyleEncoder)
    netStyleEncoder.cuda(0)

    return netStyleEncoder

def define_MappingNet():
    netMapping = MappingNetwork()
    print('netMapping:')
    print(netMapping)
    netMapping.cuda(0)

    return netMapping



def define_distan_G(input_nc, output_nc, ngf, n_downsample_global=2,
             id_enc_norm='pixel', gpu_ids=[], padding_type='reflect',
             style_dim=50, init_type='gaussian',
             conv_weight_norm=False, decoder_norm='pixel', activation='lrelu',
             adaptive_blocks=4, normalize_mlp=False, modulated_conv=False):

    id_enc_norm = get_norm_layer(norm_type=id_enc_norm)

    netG = Distan_Generator(input_nc, output_nc, ngf, n_downsampling=n_downsample_global,
                     id_enc_norm=id_enc_norm, padding_type=padding_type, style_dim=style_dim,
                     conv_weight_norm=conv_weight_norm, decoder_norm=decoder_norm,
                     actvn=activation, adaptive_blocks=adaptive_blocks,
                     normalize_mlp=normalize_mlp, modulated_conv=modulated_conv)
 
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])

    netG.apply(weights_init(init_type))

    return netG

def define_D(input_nc, ndf, n_layers=6, numClasses=2, gpu_ids=[],
             init_type='gaussian'):

    netD = StyleGANDiscriminator(input_nc, ndf=ndf, n_layers=n_layers,
                                 numClasses=numClasses)

    # print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])

    netD.apply(weights_init('gaussian'))

    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Data parallel wrapper
##############################################################################
class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            print(name)
            return getattr(self.module, name)


##############################################################################
# Losses
##############################################################################
class FeatureConsistency(nn.Module):
    def __init__(self):
        super(FeatureConsistency, self).__init__()

    def __call__(self,input,target):
        return torch.mean(torch.abs(input - target))


class R1_reg(nn.Module):
    def __init__(self, lambda_r1=10.0):
        super(R1_reg, self).__init__()
        self.lambda_r1 = lambda_r1

    def __call__(self, d_out, d_in):
        """Compute gradient penalty: (L2_norm(dy/dx))**2."""
        b = d_in.shape[0]
        dydx = torch.autograd.grad(outputs=d_out.mean(),
                                   inputs=d_in,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx_sq = dydx.pow(2)
        assert (dydx_sq.size() == d_in.size())
        r1_reg = dydx_sq.sum() / b

        return r1_reg * self.lambda_r1


class SelectiveClassesNonSatGANLoss(nn.Module):
    def __init__(self):
        super(SelectiveClassesNonSatGANLoss, self).__init__()
        self.sofplus = nn.Softplus()

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        bSize = input.shape[0]
        b_ind = torch.arange(bSize).long()  # tensor([ 0,  1])
        # target_classes  [3,5]
        relevant_inputs = input[b_ind, target_classes, :, :]
        # print(relevant_inputs.shape)  torch.Size([2, 1, 1])
        if target_is_real:
            loss = self.sofplus(-relevant_inputs).mean()
        else:
            loss = self.sofplus(relevant_inputs).mean()

        return loss

##############################################################################
# Generator
##############################################################################
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)

class ModulatedConv2d(nn.Module):
    def __init__(self, fin, fout, kernel_size, padding_type='reflect', upsample=False, downsample=False, latent_dim=256, normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        padding_size = kernel_size // 2
        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True


        self.weight = nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))
        self.conv = F.conv2d

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, fin), PixelNorm())

        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        self.blur = Blur(fout)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)

        if self.upsample:
            self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        if self.downsample:
            self.downsampler = nn.AvgPool2d(2)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
     
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
   
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

   
        s = 1 + self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
     
        weight = s * weight
     
        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = self.upsampler(input)

        if self.downsample:
            input = self.blur(input)


        b,_,h,w = input.shape
        # print(input.shape)  # torch.Size([6, 256, 64, 64])    train: torch.Size([2, 256, 64, 64])
        # print('aa')
        input = input.view(1,-1,h,w)
        input = self.padding(input)  
  
        out = self.conv(input, weight, groups=b).view(b, self.out_channels, h, w) + self.bias

        if self.downsample:
            out = self.downsampler(out)

        if self.upsample:
            out = self.blur(out)

        return out

class Modulated_1D(nn.Module):
    def __init__(self, in_channel, latent_dim, normalize_mlp=False):
        super().__init__()

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, in_channel), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, in_channel)

    def forward(self, x, latent):

        #import ipdb; ipdb.set_trace()

        s = 1 + self.mlp_class_std(latent)
        x = x * s
        d = torch.rsqrt((x ** 2).sum(1)+1e-5).view(-1,1)#.detach()
        x = x * d
        return x
        


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc,
                 weight_norm=False, activation='relu', normalize_mlp=False):#, pixel_norm=False):
        super(MLP, self).__init__()
        if weight_norm:
            linear = EqualLinear
        else:
            linear = nn.Linear

        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        elif activation == 'blrelu':
            actvn = BidirectionalLeakyReLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.model = []

        # normalize input
        if normalize_mlp:
            self.model += [PixelNorm()]

         # set the first layer
        self.model += [linear(input_dim, fc_dim),
                       actvn]
        if normalize_mlp:
            self.model += [PixelNorm()]

        # set the inner layers
        for i in range(n_fc - 2):
            self.model += [linear(fc_dim, fc_dim),
                           actvn]
            if normalize_mlp:
                self.model += [PixelNorm()]

        # set the last layer
        self.model += [linear(fc_dim, out_dim)] # no output activations

        # normalize output
        if normalize_mlp:
            self.model += [PixelNorm()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.model(input)
        return out

class StyledConvBlock(nn.Module):
    def __init__(self, fin, fout, latent_dim=256, padding='reflect', upsample=False, downsample=False,
                 actvn='lrelu', use_pixel_norm=False, normalize_affine_output=False, modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.use_pixel_norm = use_pixel_norm
        self.upsample = upsample
        self.downsample = downsample
        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2,True)

        if self.downsample:
            self.downsampler = nn.AvgPool2d(2)

        if self.modulated_conv:
            self.conv0 = conv2d(fin, fout, kernel_size=3, padding_type=padding, upsample=upsample,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)
            if self.upsample:
                seq0 = [self.upsampler, padding_layer(1), conv0, Blur(fout)]
            else:
                seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        if use_pixel_norm:
            self.pxl_norm0 = PixelNorm()

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(fout, fout, kernel_size=3, padding_type=padding, downsample=downsample,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            if self.downsample:
                seq1 = [Blur(fout), padding_layer(1), conv1, self.downsampler]
            else:
                seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        if use_pixel_norm:
            self.pxl_norm1 = PixelNorm()

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain
        if self.use_pixel_norm:
            out = self.pxl_norm0(out)

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain
        if self.use_pixel_norm:
            out = self.pxl_norm1(out)

        return out

class IdentityEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=7,
                 norm_layer=PixelNorm, padding_type='reflect',
                 conv_weight_norm=False, actvn='relu'):
        assert(n_blocks >= 0)
        super(IdentityEncoder, self).__init__()

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        encoder = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)]

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        return self.encoder(input)

class Distan_Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=7,
                 norm_layer=PixelNorm, padding_type='reflect',
                 conv_weight_norm=False, actvn='relu'):
        assert(n_blocks >= 0)
        super(Distan_Encoder, self).__init__()

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        encoder = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        ###
        for i in range(n_blocks-1):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)]

        self.encoder = nn.Sequential(*encoder)

        self.id_layer = ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)

        self.structure_layer = ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)
        #self.norm_struct = nn.LayerNorm()
        text = [padding_layer(1), conv2d(256, 512, kernel_size=3, stride=2, padding=0),norm_layer(ngf * mult * 2), activation]
        text += [padding_layer(1), conv2d(512, 1024, kernel_size=3, stride=2, padding=0),norm_layer(ngf * mult * 2), activation]
        text += [nn.AdaptiveAvgPool2d(1)]
        text += [conv2d(1024, 256, kernel_size=1, stride=1, padding=0)]
        self.text_layer = nn.Sequential(*text)
        
        #self.tex_layer = nn.Sequential(
        #    ConvLayer(ngf * mult, ngf * mult * 2, 3, downsample=True, padding="valid"),
        #    ConvLayer(ngf * mult * 2, ngf * mult * 4, 3, downsample=True, padding="valid"),
        #    nn.AdaptiveAvgPool2d(1),
        #    ConvLayer(ngf * mult * 4, 256, 1),
        #)

    def forward(self, input):
        feat = self.encoder(input)
        id_feat = self.id_layer(feat)
        structure = self.structure_layer(feat)
        texture = self.text_layer(feat)
        return id_feat, structure, texture


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

# 1 加入了 stargan v2中的 style encoder fig2 (c)
class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=256, num_domains=6, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size  # 64
        blocks = []
        # 1
        # blocks += [PixelNorm()]

        # in_channels  out_channels  kernel_size  stride  padding
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        # print('repeat_num: ' + str(repeat_num))    6

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)


        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]  # 几个域 对应几个 全连接层输出
            # 1
            # self.unshared += [PixelNorm()]

    def forward(self, x, class_A=None, class_B=None):
       
        
        h = self.shared(x)  # torch.Size([512, 1, 1])
        # h = torch.squeeze(h)
        # print(h.shape)   #num_domains=1  torch.Size([2, 512, 1, 1])   num_domains=6   [2, 512, 1, 1]
        h = h.view(h.size(0), -1)
        # print(h.shape)  torch.Size([2, 512])

        out = []
        for layer in self.unshared:
            out += [layer(h)]
        # print(len(out))  6
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # print(out.shape)  #num_domains=6 torch.Size([2, 6, 256])   num_domains=1 torch.Size([2, 1, 256])
       
        # print(out[0][class_A].shape) #num_domains=6  torch.Size([1, 256])

        out = torch.stack((out[0][class_A][0], out[1][class_B][0]))  code
    # num_domains=1:   
        # out = torch.squeeze(out)

        # print(out.shape)   # num_domains=6 or 1   torch.Size([2, 256])
       
      
        return out
# end 1111111111111111111111111111111

# 1  v2 mapping net
class MappingNetwork(nn.Module):
   
    def __init__(self, latent_dim=16, style_dim=256, num_domains=6):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(5):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))] #  一共8层

    def forward(self, z, class_A=None, class_B=None, mode='train', infer=False):
        h = self.shared(z)
        # print(h.shape)  #torch.Size([2, 512])
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # print(out.shape)  #torch.Size([2, 6, 256])        'test'  torch.Size([1, 6, 256])
         
      
        if mode == 'test':
            # s = torch.stack((out[0:500][class_A], out[500:1000][class_B]))
            # s = [out[:, class_A, :], out[:, class_B, :]]
            s = out[0][class_A]
            # print(s.shape)   # torch.Size([1, 256])
            return s
        
        if infer:
            s = [out[:, class_A, :], out[:, class_B, :]]
        else:
            s = torch.stack((out[0][class_A][0], out[1][class_B][0]))
        # s = out[idx, y]  # (batch, style_dim)
        # print(s.shape)   torch.Size([2, 256])
    
        return s
#  Mapping net  end





    

class Distan_Encoder_1(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=7,
                 norm_layer=PixelNorm, padding_type='reflect',
                 conv_weight_norm=False, actvn='relu'):
        assert(n_blocks >= 0)
        super(Distan_Encoder_1, self).__init__()

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        encoder1 = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder1 += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        norm_layer(ngf * mult * 2), activation]
        self.encoder1 = nn.Sequential(*encoder1)

        ### resnet blocks
        mult = 2**n_downsampling
        ###
        encoder2 = []
        for i in range(n_blocks-1):
            encoder2 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)]

        self.encoder2 = nn.Sequential(*encoder2)

        id_layer = [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)]
        id_layer += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        norm_layer(ngf * mult * 2), activation]
        self.id_layer = nn.Sequential(*id_layer)

        self.structure_layer = ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)
        #self.norm_struct = nn.LayerNorm()
        text = [padding_layer(1), conv2d(256, 512, kernel_size=3, stride=2, padding=0),norm_layer(ngf * mult * 2), activation]
        text += [padding_layer(1), conv2d(512, 1024, kernel_size=3, stride=2, padding=0),norm_layer(ngf * mult * 2), activation]
        text += [nn.AdaptiveAvgPool2d(1)]
        text += [conv2d(1024, 256, kernel_size=1, stride=1, padding=0)]
        self.text_layer = nn.Sequential(*text)
        
        #self.tex_layer = nn.Sequential(
        #    ConvLayer(ngf * mult, ngf * mult * 2, 3, downsample=True, padding="valid"),
        #    ConvLayer(ngf * mult * 2, ngf * mult * 4, 3, downsample=True, padding="valid"),
        #    nn.AdaptiveAvgPool2d(1),
        #    ConvLayer(ngf * mult * 4, 256, 1),
        #)

    def forward(self, input):
        #import ipdb; ipdb.set_trace()
        feat1 = self.encoder1(input)
        structure = self.structure_layer(feat1)
        feat2 = self.encoder2(feat1)
        id_feat = self.id_layer(feat2)
        #structure = self.structure_layer(feat2)
        texture = self.text_layer(feat2)
        return id_feat, structure, texture 

class AgeEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=4, style_dim=50, padding_type='reflect',
                 conv_weight_norm=False, actvn='lrelu'):
        super(AgeEncoder, self).__init__()

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        encoder = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        activation]

        encoder += [conv2d(ngf * mult * 2, style_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        features = self.encoder(input)
        latent = features.mean(dim=3).mean(dim=2)
        return latent

class StyledDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, style_dim=50, latent_dim=256, n_downsampling=2,
                 padding_type='reflect', actvn='lrelu', use_tanh=True, use_pixel_norm=False,
                 normalize_mlp=False, modulated_conv=False):
        super(StyledDecoder, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        mult = 2**n_downsampling
        last_upconv_out_layers = ngf * mult // 4

        self.StyledConvBlock_0 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_1 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_2 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_3 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_up0 = StyledConvBlock(ngf * mult, ngf * mult // 2, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)
        self.StyledConvBlock_up1 = StyledConvBlock(ngf * mult // 2, last_upconv_out_layers, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)

        self.conv_img = nn.Sequential(EqualConv2d(last_upconv_out_layers, output_nc, 1), nn.Tanh())
        self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=True, activation=actvn, normalize_mlp=normalize_mlp)

    def forward(self, id_features, target_age=None, traverse=False, deploy=False, interp_step=0.5):

        #import ipdb; ipdb.set_trace()
        if target_age is not None:
            if traverse:
                # tensor([[1.0000],
                #        [0.5000]])
                alphas = torch.arange(1,0,step=-interp_step).view(-1,1).cuda()
                interps = len(alphas)
                orig_class_num = target_age.shape[0]
                output_classes = interps * (orig_class_num - 1) + 1
                temp_latent = self.mlp(target_age)    
                latent = temp_latent.new_zeros((output_classes, temp_latent.shape[1]))
            else:
                latent = self.mlp(target_age)
        else:
            latent = None

        if traverse:
            id_features = id_features.repeat(output_classes,1,1,1)
            for i in range(orig_class_num-1):
                latent[interps*i:interps*(i+1), :] = alphas * temp_latent[i,:] + (1 - alphas) * temp_latent[i+1,:]
            latent[-1,:] = temp_latent[-1,:]
        elif deploy:
            output_classes = target_age.shape[0]
            id_features = id_features.repeat(output_classes,1,1,1)

        out = self.StyledConvBlock_0(id_features, latent)
        out = self.StyledConvBlock_1(out, latent)
        out = self.StyledConvBlock_2(out, latent)
        out = self.StyledConvBlock_3(out, latent)
        out = self.StyledConvBlock_up0(out, latent)
        out = self.StyledConvBlock_up1(out, latent)
        out = self.conv_img(out)

        return out

class Distan_StyledDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, style_dim=50, latent_dim=256, n_downsampling=2,
                 padding_type='reflect', actvn='lrelu', use_tanh=True, use_pixel_norm=False,
                 normalize_mlp=False, modulated_conv=False):
        super(Distan_StyledDecoder, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        mult = 2**n_downsampling
        last_upconv_out_layers = ngf * mult // 4

        self.StyledConvBlock_0 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_1 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_2 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_3 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_up0 = StyledConvBlock(ngf * mult, ngf * mult // 2, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)
        self.StyledConvBlock_up1 = StyledConvBlock(ngf * mult // 2, last_upconv_out_layers, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)

        self.conv_img = nn.Sequential(EqualConv2d(last_upconv_out_layers, output_nc, 1), nn.Tanh())
        self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=True, activation=actvn, normalize_mlp=normalize_mlp)
        #self.t_denorm = nn.Sequential(nn.Linear(256*2,256*4),nn.LeakyReLU(0.2, True), nn.Linear(256*4,256))

        self.s_transform = ModulatedConv2d(256, 256, kernel_size=3, padding_type=padding_type, upsample=False,
                                           latent_dim=256, normalize_mlp=normalize_mlp)

        self.t_transform = Modulated_1D(256,256,normalize_mlp=normalize_mlp)
        #self.s_denorm = nn.Linear(256,2*256)
        self.t_denorm = nn.Linear(256,2)

    def forward(self, struct_feat, text_feat, target_age=None, traverse=False, deploy=False, interp_step=0.5, latent_in=None):

        #import ipdb; ipdb.set_trace()
        # if target_age is not None:
        if latent_in is not None:
            if traverse: 
                alphas = torch.arange(1,0,step=-interp_step).view(-1,1).cuda()
                interps = len(alphas)
                orig_class_num = target_age.shape[0]
                output_classes = interps * (orig_class_num - 1) + 1
                temp_latent = self.mlp(target_age)
                latent = temp_latent.new_zeros((output_classes, temp_latent.shape[1]))
            else:
                # 1 2/7  latent = self.mlp(target_age)
                if latent_in is None:
                    # print(target_age.shape) # torch.Size([2, 300])
                    latent = self.mlp(target_age)
                    # print(latent.shape)   #torch.Size([2, 256])
                else:
                    latent = latent_in

        else:
            latent = None

        if traverse:
            #id_features = id_features.repeat(output_classes,1,1,1)
            struct_feat = struct_feat.repeat(output_classes, 1,1,1)
            text_feat = text_feat.repeat(output_classes, 1,1,1)
            for i in range(orig_class_num-1):
                latent[interps*i:interps*(i+1), :] = alphas * temp_latent[i,:] + (1 - alphas) * temp_latent[i+1,:]
            latent[-1,:] = temp_latent[-1,:]
        elif deploy:
            output_classes = target_age.shape[0]
            struct_feat = struct_feat.repeat(output_classes, 1,1,1)
            text_feat = text_feat.repeat(output_classes, 1,1,1)
        ##norm and denorm if features
        ##normalizing structure
        if latent_in is not None:
            B, C, W, H = struct_feat.size()
            
        ##normalizing structure
            if latent_in is not None:
                latent = latent_in
             


    # print(text_feat.shape, latent.shape)#torch.Size([0, 256, 1, 1]) torch.Size([2, 256])
            # print(struct_feat.shape)  torch.Size([0, 256, 64, 64])
            new_struct = self.s_transform(struct_feat, latent)
            
        ###normalizing texture
            text_feat = text_feat.contiguous().reshape(B,C)

           
            new_text = self.t_transform(text_feat, latent)
            
          
        else:
            B, C, W, H = struct_feat.size()
            new_struct = struct_feat
            new_text = text_feat.contiguous().reshape(B,C)
        
        #id_features = id
        ##norm and denorm latent features
        # m = int(np.random.randint(2, size=1))
        # n = int(np.random.randint(256, size=1))
        out = self.StyledConvBlock_0(new_struct, new_text)
        # new_struct = new_struct[m][n][0:8].reshape(2,256)
        # new_struct = new_struct[m][n][0:24].reshape(6, 256)
        out = self.StyledConvBlock_1(out, new_text)
        out = self.StyledConvBlock_2(out, new_text)
        out = self.StyledConvBlock_3(out, new_text)
        out = self.StyledConvBlock_up0(out, new_text)
        out = self.StyledConvBlock_up1(out, new_text)
        out = self.conv_img(out)

        return out, new_struct, new_text, latent

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=50, n_downsampling=2,
                 n_blocks=4, adaptive_blocks=4, id_enc_norm=PixelNorm,
                 padding_type='reflect', conv_weight_norm=False,
                 decoder_norm='pixel', actvn='lrelu', normalize_mlp=False,
                 modulated_conv=False):
        super(Generator, self).__init__()
        self.id_encoder = IdentityEncoder(input_nc, ngf, n_downsampling, n_blocks, id_enc_norm,
                                          padding_type, conv_weight_norm=conv_weight_norm,
                                          actvn='relu') # replacing relu with leaky relu here causes nans and the entire training to collapse immediately
        self.age_encoder = AgeEncoder(input_nc, ngf=ngf, n_downsampling=4, style_dim=style_dim,
                                      padding_type=padding_type, actvn=actvn,
                                      conv_weight_norm=conv_weight_norm)

        use_pixel_norm = decoder_norm == 'pixel'
        self.decoder = StyledDecoder(output_nc, ngf=ngf, style_dim=style_dim,
                                     n_downsampling=n_downsampling, actvn=actvn,
                                     use_pixel_norm=use_pixel_norm,
                                     normalize_mlp=normalize_mlp,
                                     modulated_conv=modulated_conv)

    def encode(self, input):
        if torch.is_tensor(input):
            id_features = self.id_encoder(input)
            age_features = self.age_encoder(input)
            return id_features, age_features
        else:
            return None, None

    def decode(self, id_features, target_age_features, traverse=False, deploy=False, interp_step=0.5):
        if torch.is_tensor(id_features):
            return self.decoder(id_features, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        else:
            return None

    #parallel forward
    def forward(self, input, target_age_code, cyc_age_code, source_age_code, disc_pass=False):
        orig_id_features = self.id_encoder(input)
        orig_age_features = self.age_encoder(input)
        if disc_pass:
            rec_out = None
        else:
            rec_out = self.decode(orig_id_features, source_age_code)

        gen_out = self.decode(orig_id_features, target_age_code)
        if disc_pass:
            fake_id_features = None
            fake_age_features = None
            cyc_out = None
        else:
            fake_id_features = self.id_encoder(gen_out)
            fake_age_features = self.age_encoder(gen_out)
            cyc_out = self.decode(fake_id_features, cyc_age_code)
        return rec_out, gen_out, cyc_out, orig_id_features, orig_age_features, fake_id_features, fake_age_features


    def infer(self, input, target_age_features, traverse=False, deploy=False, interp_step=0.5):
        id_features = self.id_encoder(input)
        out = self.decode(id_features, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        return out


class StyleEncoder_L(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, style_dim=64, num_domains=6, max_conv_dim=512):
        super().__init__()
        blocks = []
        blocks += [nn.Linear(input_dim, hidden_dim)]
        blocks += [nn.LeakyReLU(0.2)]
        
        for _ in range(5):
            blocks += [nn.Linear(hidden_dim, hidden_dim)]
            blocks += [nn.LeakyReLU(0.2)]
            blocks += [nn.Linear(hidden_dim, hidden_dim)]
            blocks += [nn.LeakyReLU(0.2)]

        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()

        for _ in range(num_domains):
            self.unshared += [nn.Linear(hidden_dim, style_dim)]  # 几个域 对应几个 全连接层输出
       
    def forward(self, x, class_A=None):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        # print(h.shape)  # torch.Size([1, 128])
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        # print(out.shape)  # torch.Size([1, 6, 64])
        out = out[0][class_A][0]
        # print(out.shape)  #torch.Size([64])
        return out


class MappingNetwork_L(nn.Module):
  
    def __init__(self, latent_dim=16, style_dim=64, num_domains=6):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))] #  一共8层
    
    def forward(self, z, class_A=None, class_B=None, infer=None):
        h = self.shared(z)
        # print(h.shape)  torch.Size([2, 512])
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # print(out.shape)    #torch.Size([2, 6, 64])

        if infer:
            s = [out[:, class_A, :], out[:, class_B, :]]
        else:
            s = torch.stack((out[0][class_A][0], out[1][class_B][0]))
        # s = out[idx, y]  # (batch, style_dim)
        # print(s.shape)    #torch.Size([2, 64])
    
        return s
#  ###################################################





def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),
                                  
                                    nn.Linear(map_hidden_dim, map_output_dim))
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts
    

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, freq, phase_shift):
        x = x.to(torch.float32)
        x = self.layer(x)  
        # print(freq.unsqueeze(1).shape)  # [256, 1]
        # print(x.shape) #[1, 256]
        # print(phase_shift.unsqueeze(1).shape) #[256, 1]
        freq = torch.t(freq.unsqueeze(1)).expand_as(x)
        phase_shift = torch.t(phase_shift.unsqueeze(1)).expand_as(x)
        return self.LeakyReLU(freq * x + phase_shift)
    
# Primary
class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=32, z_dim=64, hidden_dim=256, output_dim=32, device=None):
        super().__init__()
        self.device = device  
        self.input_dim = input_dim  
        self.z_dim = z_dim  
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim   

        self.PN = PixelNorm()
        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, output_dim),  
            FiLMLayer(hidden_dim, hidden_dim),
        ])
    
        self.final_layer = nn.Linear(hidden_dim, output_dim)

        self.mapping_network = CustomMappingNetwork(z_dim, 512, (len(self.network) + 1)*hidden_dim*2)

        # self.network.apply(frequency_init(25))
        self.network.apply(frequency_init(10))
        
        # self.final_layer.apply(frequency_init(25))
       
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
       
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, **kwargs):
        
        frequencies = frequencies

        x = input

        for index, layer in enumerate(self.network):
            if index == 9:
                start = (index-1)* self.hidden_dim + self.output_dim
                end = (index-1)* self.hidden_dim + self.output_dim + self.output_dim
            else:
                start = index * self.hidden_dim
                end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        landmark_trans = self.final_layer(x)
        # print(landmark_trans.shape)  torch.Size([1, 32])
        # landmark_trans = self.PN(landmark_trans)
        # landmark_trans = x
       

        return landmark_trans


##disentangled generator with disentagled id-encoder and disentangled face aging

class Distan_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=50, n_downsampling=2,
                 n_blocks=4, adaptive_blocks=4, id_enc_norm=PixelNorm,
                 padding_type='reflect', conv_weight_norm=False,
                 decoder_norm='pixel', actvn='lrelu', normalize_mlp=False,
                 modulated_conv=False):
        super(Distan_Generator, self).__init__()
        # 1  加入了style encoder
        # self.style_encoder = StyleEncoder()

        self.id_encoder = Distan_Encoder_1(input_nc, ngf, n_downsampling, n_blocks, id_enc_norm,
                                          padding_type, conv_weight_norm=conv_weight_norm,
                                          actvn='relu')
        self.age_encoder = AgeEncoder(input_nc, ngf=ngf, n_downsampling=4, style_dim=style_dim,
                                      padding_type=padding_type, actvn=actvn,
                                      conv_weight_norm=conv_weight_norm)

        use_pixel_norm = decoder_norm == 'pixel'
        self.decoder = Distan_StyledDecoder(output_nc, ngf=ngf, style_dim=style_dim,
                                     n_downsampling=n_downsampling, actvn=actvn,
                                     use_pixel_norm=use_pixel_norm,
                                     normalize_mlp=normalize_mlp,
                                     modulated_conv=modulated_conv)

    def encode(self, input):
        if torch.is_tensor(input):
            id_features, struct_features, text_features = self.id_encoder(input)
            age_features = self.age_encoder(input)
            return id_features, struct_features, text_features, age_features
        else:
            return None, None, None

    def decode(self, struct_features, text_features, target_age_features=None, traverse=False, deploy=False, interp_step=0.5, latent=None):
        if torch.is_tensor(struct_features):
            return self.decoder(struct_features, text_features, target_age=target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step, latent_in=latent )
        else:
            return None

    # def styleEncode(self, input):

    #     style = self.style_encoder(input)
    #     return style


    def rec(self, input ,target_age_code, style=None):
        orig_id_features, orig_structure_feat, orig_texture_feat = self.id_encoder(input)
        if style is None:
            input_style = self.style_encoder(input)  

        else:
            input_style = style


        rec_out, _, _, _ = self.decode(orig_structure_feat, orig_texture_feat, target_age_features=target_age_code,
                                       latent=input_style)
        return rec_out

    def forward(self, input, StyleCode, disc_pass=False):
        #import ipdb; ipdb.set_trace()
        id_features, structure_feat, texture_feat = self.id_encoder(input)
        #swap_id_features, swap_structure_feat, swap_texture_feat = self.id_encoder(input_swap)
        orig_age_features = self.age_encoder(input)

     
        gen_out, new_struc, new_text, ori_latent = self.decode(structure_feat, texture_feat, target_age_features = None, latent=StyleCode)
      
        # if disc_pass:
        #     rec_out = None
        #     #distan_out = None
        # else:
        #     rec_out, _, _, _ = self.decode(structure_feat, texture_feat, target_age_features = None, latent=StyleCode)

        return gen_out, id_features, structure_feat, texture_feat, \
               orig_age_features, new_struc, new_text, ori_latent


    # def infer(self, input, target_age_features, traverse=False, deploy=False, interp_step=0.5, style=None):
    def infer(self, input, StyleCode, traverse=False, deploy=False, interp_step=0.5):
     
        id_features, structure_feat, texture_feat = self.id_encoder(input)
        # out = self.decode(structure_feat, texture_feat, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step, latent=style)
        out, new_struc, new_text, ori_latent = self.decode(structure_feat, texture_feat, target_age_features = None, latent=StyleCode)
        
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True),
                 conv_weight_norm=False, use_pixel_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation,
                                                conv_weight_norm, use_pixel_norm)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, conv_weight_norm, use_pixel_norm):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        self.use_pixel_norm = use_pixel_norm
        if self.use_pixel_norm:
            self.pixel_norm = PixelNorm()

        conv_block += [conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

##############################################################################
# Discriminator
##############################################################################
class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, numClasses=2, padding_type='reflect'):
        super(StyleGANDiscriminator, self).__init__()
        self.n_layers = n_layers
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        activation = nn.LeakyReLU(0.2,True)

        sequence = [padding_layer(0), EqualConv2d(input_nc, ndf, kernel_size=1), activation]

        nf = ndf
        for n in range(n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [StyledConvBlock(nf_prev, nf, downsample=True, actvn=activation)]

        self.model = nn.Sequential(*sequence)

        output_nc = numClasses

        self.gan_head = nn.Sequential(padding_layer(1), EqualConv2d(nf+1, nf, kernel_size=3), activation,
                                      EqualConv2d(nf, output_nc, kernel_size=4), activation)

    def minibatch_stdev(self, input):
        out_std = torch.sqrt(input.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(input.size(0), 1, input.size(2), input.size(3))
        out = torch.cat((input, mean_std), 1)
        return out

    def forward(self, input):
        features = self.model(input)
        # print(features.shape)   torch.Size([2, 512, 4, 4])
        gan_out = self.gan_head(self.minibatch_stdev(features))
        # print(gan_out.shape) [2, 6, 1, 1]
        return gan_out
