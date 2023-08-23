import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import torch
from models.model_parts import *


class Generator_shallow(nn.Module):

    def __init__(self, input_nc, output_nc, args, num_pool_layers=2, n_residual_blocks=9):

        super(Generator_shallow, self).__init__()

        self.imsize = args.size
        self.in_chans = input_nc
        self.out_chans = output_nc
        self.num_pool_layers = num_pool_layers
        self.n_residual_blocks = n_residual_blocks

        in_features = 32
        self.en_layer1 = nn.Sequential(ConvBlockFirst(1, in_features), ConvBlock(in_features, in_features), ConvBlock(in_features, in_features))     
        out_features = in_features*2
        self.down_sample_layer1 = nn.Sequential(DownConvBlock(in_features, out_features))
        
     
        out_features_ori = out_features
        self.patch_size= 2
        self.img_size=  self.imsize // 2

        ########### You better modulate the network here #######

        parser = argparse.ArgumentParser()
        self.args_ = parser.parse_args()
        self.args_.dims              = (128,)
        self.args_.heads             = (8,)
        self.args_.ff_expansion      = (4,)
        self.args_.reduction_ratio   = (2,)
        self.args_.num_layers        = (4,)
        self.args_.stage_kernel_stride_pad= ((3,2,1),)
        
        
        self.IISA = IISA(self.args_,out_features)
        
        
        current_size = self.img_size // 2**(len(self.args_.dims))

        in_features=  self.args_.dims[-1]

        self.upsample_layers = []
        while current_size < self.img_size : 
            out_features = in_features//2
            self.upsample_layers.append(nn.Sequential(TransposeConvBlock(in_features, out_features)))
            in_features = out_features
            current_size *= 2

        # Fit the last layer to the desired output size
        self.upsample_layers = nn.ModuleList(self.upsample_layers)


        self.up_sample_layer1 = nn.Sequential(TransposeConvBlock(out_features, out_features//2))
        self.de_layer1 = nn.Sequential(ConvBlock(out_features//2+out_features_ori//2, out_features_ori),ConvBlock(out_features_ori, out_features_ori))


        in_features = out_features_ori
        out_features = in_features//2
        
        self.last_conv = nn.Sequential(ConvBlockLast(in_features, 1))

    def forward(self, input):

        input = input.permute(1,0,2,3).contiguous()
        
        stack = []        
        output = input
        
        output = self.en_layer1(output)
        stack.append(output)
        output = self.down_sample_layer1(output)

        output = self.IISA(output)
        output_img = output

        for layers in self.upsample_layers : 
            output_img = layers(output_img)

        d_sample_layer = stack.pop()

        output_img = self.up_sample_layer1(output_img)

        #################### Here for cat ####################
        output_img = torch.cat([output_img, d_sample_layer], dim=1)
        
        output_img = self.de_layer1(output_img)
        output_img = self.last_conv(output_img)
        output_img = output_img.permute(1,0,2,3).contiguous()
        
        return output_img


      
      
class Discriminator(nn.Module):
    def __init__(self, in_channels, patch_size=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(  nn.Conv2d(1, 64  , kernel_size=4, stride=2), nn.InstanceNorm2d(64), nn.LeakyReLU(negative_slope=0.2, inplace=True) )
        self.conv2 = nn.Sequential(  nn.Conv2d(64,128 , kernel_size=4, stride=2), nn.InstanceNorm2d(128), nn.LeakyReLU(negative_slope=0.2, inplace=True) )
        self.conv3 = nn.Sequential(  nn.Conv2d(128,256, kernel_size=4, stride=2), nn.InstanceNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True) )
        self.conv4 = nn.Sequential(  nn.Conv2d(256,512, kernel_size=4, stride=2), nn.InstanceNorm2d(512), nn.LeakyReLU(negative_slope=0.2, inplace=True) )
        self.conv5 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        

    def forward(self, x):
        
        x= x.permute(1,0,2,3).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return torch.mean(x, [-2,-1])
    
    
    
    
    
