
import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, input_channels, filters, n_downsampling_layers, n_resnet_layers):
        super(Encoder, self).__init__()
        model = []
        model += [ConvolutionalBlock(input_channels, filters, 7, 1, 3, 'in', 'relu', 'reflection')]
    
        for _ in range(n_downsampling_layers):
            model += [ConvolutionalBlock(filters, 2 * filters, 4, 2, 1, 'in', 'relu', 'reflection')]
            filters *= 2
        self.output_channels = filters
        for _ in range(n_resnet_layers):
            model += [ResNetBlock(filters, 'in', 'relu', 'reflection')]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, n_upsampling_layers, n_resnet_layers):
        super(Decoder, self).__init__()
        model = []
        for _ in range(n_resnet_layers):
            model += [ResNetBlock(input_channels, 'in', 'relu', 'reflection')]
        filters = input_channels
        for _ in range(n_upsampling_layers):
            model += [nn.Upsample(scale_factor = 2), ConvolutionalBlock(filters, filters // 2, 5, 1, 2, 'in', 'relu', 'reflection')]
            filters = filters // 2
        model += [ConvolutionalBlock(filters, output_channels, 7, 1, 3, 'none', 'tanh', 'reflection')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
        
class ResNetBlock(nn.Module):
    def __init__(self, channels, norm, activation, pad_type):
        super(ResNetBlock, self).__init__()
        model = []
        model += [ConvolutionalBlock(channels, channels, 3, 1, 1, norm, activation, pad_type)]
        model += [ConvolutionalBlock(channels, channels, 3, 1, 1, norm, 'none', pad_type)]

        self.model = nn.Sequential(*model)
    def forward(self, x):
        residual_output = x
        output = self.model(x)

        return output + residual_output


class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, norm = 'none', activation = 'relu', pad_type = 'zero'):
        super(ConvolutionalBlock, self).__init__()
        if pad_type == 'reflection':
            self.padding_layer = nn.ReflectionPad2d(padding = padding)
        elif pad_type == 'zero':
            self.padding_layer = nn.ZeroPad2d(padding = padding)
        self.conv_layer = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel_size, stride = stride, bias = True)
        
        if activation == 'relu':
            self.activation_layer = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation_layer = nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        elif activation == 'tanh':
            self.activation_layer = nn.Tanh()
        else:
            self.activation_layer = None
        
        if norm == 'in':
            self.normalization_layer = nn.InstanceNorm2d(num_features = output_channels)
        elif norm == 'bn':
            self.normalization_layer = nn.BatchNorm2d(num_features = output_channels)
        elif norm == 'ln':
            self.normalization_layer = nn.LayerNorm(normalized_shape = output_channels)
        else:
            self.normalization_layer = None

    def forward(self, x):
        x = self.conv_layer(self.padding_layer(x))

        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        return x


