
import torch.nn as nn
from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ConvBlockFirst(nn.Module):

    def __init__(self, in_chans, out_chans):

        super(ConvBlockFirst, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.in_chans, self.out_chans, 7),
            nn.InstanceNorm2d(self.out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.layers(input)
    
    
    
class ConvBlockLast(nn.Module):

    def __init__(self, in_chans, out_chans):
        super(ConvBlockLast, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.in_chans, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.out_chans, 1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)
    
    
class ConvBlockLastSeg(nn.Module):

    def __init__(self, in_chans, out_chans):

        super(ConvBlockLastSeg, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.in_chans, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.out_chans, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.layers(input)
    
class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):

        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.layers(input)
    
class DownConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):

        super(DownConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_chans, self.out_chans, 3, stride=2, padding=1),
            nn.InstanceNorm2d(self.out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.layers(input)
    
class TransposeConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):

        super(TransposeConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.in_chans, self.out_chans, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(self.out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.layers(input)
    

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))



class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)
        
        
        self.to_q_b   = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv_b  = nn.Conv2d(dim, dim * 2, 1, stride=1, bias=False)
        

    def forward(self, x):
        b, _, h, w = x.shape
        heads = self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        
        
        # Inter slice
        q_img, k_img, v_img = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))
        
        sim_img = einsum('i j d, i k d -> i j k', q_img, k_img) * self.scale
        attn_img = sim_img.softmax(dim=-1)
        out_img = einsum('i j k, i k d -> i j d', attn_img, v_img)
        out_img = rearrange(out_img, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        
        
        # Intra slice
        
        q_batch, k_batch, v_batch = (self.to_q_b(x), *self.to_kv_b(x).chunk(2, dim=1))
        q_batch, k_batch, v_batch = map(lambda t: rearrange(t, 'b (h c) x y -> (x y h) b c', h=heads), (q_batch, k_batch, v_batch))
        
        sim_batch = einsum('i b d, i c d -> i b c', q_batch, k_batch) * self.scale
        attn_batch = sim_batch.softmax(dim=-1)
        out_batch = einsum('i b c, i c d -> i b d', attn_batch, v_batch)
        out_batch = rearrange(out_batch, '(x y h) b c -> b (h c) x y', h=heads, x=h, y=w)
        
        return self.to_out(out_img + out_batch)




class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers,
        stage_kernel_stride_pad,
    ):
        super().__init__()
        
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)
            layers = nn.ModuleList([])
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            
            x = get_overlap_patches(x)
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            x = overlap_embed(x)
            
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

########## THIS IS A U-NET LIKE ############

    
class IISA(nn.Module):
    def __init__(  self , args, feat_dim ):
        super().__init__()
        size_red=1
        self.args = args
        dims         = args.dims
        heads        = args.heads
        ff_expansion = args.ff_expansion
        reduction_ratio = args.reduction_ratio
        stage_kernel_stride_pad = args.stage_kernel_stride_pad
        num_layers = args.num_layers
        for k in stage_kernel_stride_pad: 
            size_red = size_red * k[1]                
        channels = feat_dim
        
                
        
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = len(dims)), (dims, heads, ff_expansion, reduction_ratio, num_layers))        
 
        assert all([*map(lambda t: len(t) ==  len(dims), (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four dims-stage are allowed, all keyword arguments must be either a single value or a tuple of dim values'


        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers,
            stage_kernel_stride_pad = stage_kernel_stride_pad
        )





    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)
        Last_layer = layer_outputs[-1]
        return Last_layer
    
    

    