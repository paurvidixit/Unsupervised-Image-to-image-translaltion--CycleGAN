import torch
import torch.nn as nn
from util_networks import ConvolutionalBlock

class Discriminator(nn.Module):
    def __init__(self, input_channels, dis_features):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = dis_features, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(in_channels = dis_features, out_channels = dis_features * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(dis_features * 2),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(in_channels = dis_features * 2, out_channels = dis_features * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(dis_features * 4),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(in_channels = dis_features * 4, out_channels = dis_features * 8, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(dis_features * 8),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(in_channels = dis_features * 8, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1)

class CycleGANDiscriminator(nn.Module):
    def __init__(self, input_channels, dis_features):
        # Batch size is 1 so instance normalization is always used, Dropout is never 
        # used and Bias is always used. Following implementation details given in the CycleGAN Paper.
        super(CycleGANDiscriminator, self).__init__()
        self.network = nn.Sequential(
            # Input processing
            nn.Conv2d(in_channels = input_channels, out_channels = dis_features, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.5, inplace = True),

            # Increase number of filters gradually
            nn.Conv2d(in_channels = dis_features, out_channels = dis_features * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(num_features = dis_features * 2),
            nn.LeakyReLU(negative_slope = 0.5, inplace = True),
            nn.Conv2d(in_channels = dis_features * 2, out_channels = dis_features * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(num_features = dis_features * 4),
            nn.LeakyReLU(negative_slope = 0.5, inplace = True),
            nn.Conv2d(in_channels = dis_features * 4, out_channels = dis_features * 8, kernel_size = 4, stride = 1, padding = 1),
            nn.InstanceNorm2d(num_features = dis_features * 8),
            nn.LeakyReLU(negative_slope = 0.5, inplace = True),
            
            # Output
            nn.Conv2d(in_channels = dis_features * 8, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        )

    def forward(self, x):
        return self.network(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels, loss):
        super(MultiScaleDiscriminator, self).__init__()
        self.filters = 64
        self.input_channels = input_channels
        n_scales = 3

        self.downsample_layer = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1, count_include_pad = False)
        self.convolutional_networks = nn.ModuleList()

        for _ in range(n_scales):
            self.convolutional_networks.append(self.create_cnn())

        if loss == 'mse':
            self.loss_criterion = nn.MSELoss()
        else:
            self.loss_criterion = nn.BCELoss()

    def create_cnn(self):
        filters = self.filters 
        n_layers = 4
        model = []
        model += [ConvolutionalBlock(self.input_channels, filters, 4, 2, 1, 'none', 'lrelu', 'reflection')]

        for _ in range(n_layers - 1):
            model += [ConvolutionalBlock(filters, 2 * filters, 4, 2, 1, 'none', 'lrelu', 'reflection')]
            filters *= 2
        model += [nn.Conv2d(in_channels = filters, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)]

        model = nn.Sequential(*model)
        return model

    def forward(self, x):
        outputs = []
        for model in self.convolutional_networks:
            outputs.append(model(x))
            x = self.downsample_layer(x)
        return outputs

    def calc_discriminator_loss(self, fake_input, real_input):
        fake_output = self.forward(fake_input)
        real_output = self.forward(real_input)
        device = torch.device('cuda')

        loss = 0
        
        fake_label = 0.0
        real_label = 1.0

        for _, (fake, real) in enumerate(zip(fake_output, real_output)):
            target_real = torch.tensor(real_label)
            target_fake = torch.tensor(fake_label)
            target_real = target_real.expand_as(real).to(device)
            target_fake = target_fake.expand_as(fake).to(device)

            loss += self.loss_criterion(real, target_real) + self.loss_criterion(fake, target_fake)
    
        return loss

    def calc_generator_loss(self, fake_input):
        fake_output = self.forward(fake_input)
        loss = 0

        device = torch.device('cuda')

        real_label = 1.0

        for _, fake in enumerate(fake_output):
            target_real = torch.tensor(real_label)
            target_real = target_real.expand_as(fake).to(device)

            loss += self.loss_criterion(fake, target_real)
        return loss