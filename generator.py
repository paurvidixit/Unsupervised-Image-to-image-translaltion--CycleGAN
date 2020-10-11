import torch
import torch.nn as nn
from util_networks import ConvolutionalBlock, ResNetBlock, Decoder, Encoder

class Generator(nn.Module):
    def __init__(self, input_channels, out_channels, gen_features):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels = input_channels, out_channels = gen_features * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = gen_features * 8),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = gen_features * 8, out_channels = gen_features * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = gen_features * 4),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = gen_features * 4, out_channels = gen_features * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = gen_features * 2),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = gen_features * 2, out_channels = gen_features, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = gen_features),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = gen_features, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Tanh()   
        )
    
    def forward(self, input):
        return self.network(input)

class ResNetBlockCycleGAN(nn.Module):
    def __init__(self, num_filters):
        """ num_filters : number of filters in the convolutional layer
        """
        # Reflection padding is always used, batch size is 1 so instance normalization is always used, Dropout is never 
        # used and Bias is always used. Following implementation details given in the CycleGAN Paper.
        super(ResNetBlockCycleGAN, self).__init__()
        self.resnet_block = nn.Sequential(
            nn.ReflectionPad2d(padding = 1),
            nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = 3, padding = 0),
            nn.InstanceNorm2d(num_features = num_filters),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(padding = 1),
            nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = 3, padding = 0),
            nn.InstanceNorm2d(num_features = num_filters),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        output = self.resnet_block(x)
        return x + output # Resnet adds skip connections


class CycleGANGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, gen_features):
        """ input_channels : number of channels in the input image
            output_channels : number of channels in the output image
            gen_features : Base number of filters
        """

        # Reflection padding is always used, batch size is 1 so instance normalization is always used, Dropout is never 
        # used and Bias is always used. Following implementation details given in the CycleGAN Paper.
        super(CycleGANGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.ReflectionPad2d(padding = 3), # Input processing
            nn.Conv2d(in_channels = input_channels, out_channels = gen_features, kernel_size = 7, padding = 0),
            nn.InstanceNorm2d(num_features = gen_features),
            nn.ReLU(inplace = True),
            # Downsampling the image
            nn.Conv2d(in_channels = gen_features, out_channels = gen_features * 2, kernel_size = 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(num_features = gen_features * 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = gen_features * 2, out_channels = gen_features * 4, kernel_size = 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(num_features = gen_features * 4),
            nn.ReLU(inplace = True),

            # Resnet blocks
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),
            ResNetBlockCycleGAN(gen_features * 4),

            # Upsampling the image
            nn.ConvTranspose2d(in_channels = gen_features * 4, out_channels = gen_features * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(gen_features * 2),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = gen_features * 2, out_channels = gen_features, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(gen_features),
            nn.ReLU(inplace = True),

            # Post processing
            nn.ReflectionPad2d(padding = 3),
            nn.Conv2d(in_channels = gen_features, out_channels = output_channels, kernel_size = 7, padding = 0),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class VAEGAN(nn.Module):
    def __init__(self, input_channels, training = True):
        super(VAEGAN, self).__init__()
        filters = 64
        n_downsampling_layers = 2
        n_resnet_layers = 4
        self.training = training

        self.encoder = Encoder(input_channels, filters, n_downsampling_layers, n_resnet_layers)
        self.decoder = Decoder(self.encoder.output_channels, input_channels, n_downsampling_layers, n_resnet_layers)
    
    def forward(self, x):
        latent, noise = self.encode(x)
        if self.training:
            output = self.decode(latent + noise)
        else:
            output = self.decode(latent)
        return output, latent

    def encode(self, input):
        device = torch.device('cuda')
        latent = self.encoder(input)
        noise = torch.randn(latent.size()).to(device)
        return latent, noise

    def decode(self, latent):
        x = self.decoder(latent)
        return x

        
        
