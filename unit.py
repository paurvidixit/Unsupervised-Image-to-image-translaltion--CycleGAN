from discriminator import MultiScaleDiscriminator
from generator import VAEGAN

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

import numpy as np
from PIL import Image

import os
import glob
from torch.utils.tensorboard import SummaryWriter
from cyclegan_dataset import CycleGANDataset
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.utils as vutils

class UNITGAN:
    def __init__(self, training = True):
        self.device = torch.device('cuda')
        self.generator_A = VAEGAN(3, training).to(self.device)
        self.generator_B = VAEGAN(3, training).to(self.device)
        self.discriminator_A = MultiScaleDiscriminator(3, 'mse').to(self.device)
        self.discriminator_B = MultiScaleDiscriminator(3, 'mse').to(self.device)

        # Print the networks
        print(self.generator_A)
        print(self.generator_B)
        print(self.discriminator_A)
        print(self.discriminator_B)

        self.generator_optimizer = optim.Adam(itertools.chain(self.generator_A.parameters(), self.generator_B.parameters()), 
                        lr = 0.0001, betas = (0.5, 0.999), weight_decay = 0.0001)
        self.discriminator_optimizer = optim.Adam(itertools.chain(self.discriminator_A.parameters(), self.discriminator_B.parameters()),
                        lr = 0.0001, betas = (0.5, 0.999), weight_decay = 0.0001)

        self.generator_scheduler = optim.lr_scheduler.StepLR(self.generator_optimizer, step_size = 100000, gamma = 0.5)
        self.discriminator_scheduler = optim.lr_scheduler.StepLR(self.discriminator_optimizer, step_size = 100000, gamma = 0.5)

        self.generator_A.apply(self.init_weights_kaiming)
        self.generator_B.apply(self.init_weights_kaiming)
        self.discriminator_A.apply(self.init_weights_gaussian)
        self.discriminator_B.apply(self.init_weights_gaussian)

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Get the dataset and dataloaders
        if training:
            self.dataset = CycleGANDataset(base_dir = '/home/paurvi/CycleGAN/datasets/summer2winter_yosemite')
        else:
            self.dataset = CycleGANDataset(base_dir = '/home/paurvi/CycleGAN/datasets/summer2winter_yosemite', phase = 'test')
            self.load_networks(25)
        self.dataloader = DataLoader(self.dataset, batch_size = 1, num_workers = 2)

        # Writers to tensorboard
        self.steps = 0
        self.writer = SummaryWriter(comment = 'unit_yosemite_128')

    def init_weights_gaussian(self, m):
        """ Initialize weights for the network
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') == 0 or classname.find('Linear') == 0):
            nn.init.normal_(m.weight.data, mean = 0.0, std = 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, val = 0.0)

    def init_weights_kaiming(self, m):
        """ Initialize weights for the network
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') == 0 or classname.find('Linear') == 0):
            nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, val = 0.0)
    

    def load_networks(self, epoch):
        path = os.path.join(os.getcwd(), 'models', 'unit')

        self.generator_A.load_state_dict(torch.load('%s/generator_A_epoch_%d.pth' % (path, epoch)))
        self.generator_B.load_state_dict(torch.load('%s/generator_B_epoch_%d.pth' % (path, epoch)))
        self.discriminator_A.load_state_dict(torch.load('%s/discriminator_A_epoch_%d.pth' % (path, epoch)))
        self.discriminator_B.load_state_dict(torch.load('%s/discriminator_B_epoch_%d.pth' % (path, epoch)))
        
    
    def forward(self, image_A, image_B):
        # self.eval()
        self.generator_A.eval()
        self.generator_B.eval()
        self.discriminator_A.eval()
        self.discriminator_B.eval()
        latent_A, _ = self.generator_A.encode(image_A)
        latent_B, _ = self.generator_B.encode(image_B)
        fake_A = self.generator_A.decode(latent_B)
        fake_B = self.generator_B.decode(latent_A)

        self.generator_A.train()
        self.generator_B.train()
        self.discriminator_A.train()
        self.discriminator_B.train()
        # self.train()
        
        return fake_A, fake_B

    def optimize_generator(self, image_A, image_B, epoch, iteration):
        image_A = image_A.to(self.device)
        image_B = image_B.to(self.device)

        # Encoding
        latent_A, noise_A = self.generator_A.encode(image_A)
        latent_B, noise_B = self.generator_B.encode(image_B)

        # Decoding within domain
        fake_A_within = self.generator_A.decode(latent_A + noise_A)
        fake_B_within = self.generator_B.decode(latent_B + noise_B)

        # Decoding cross domain
        fake_A_cross = self.generator_A.decode(latent_B + noise_B)
        fake_B_cross = self.generator_B.decode(latent_A + noise_A)

        # Encode cross domain fakes
        latent_B_cross, noise_B_cross = self.generator_A.encode(fake_A_cross)
        latent_A_cross, noise_A_cross = self.generator_B.encode(fake_B_cross)

        # Reconstruct
        reconstructed_A = self.generator_A.decode(latent_A_cross + noise_A_cross)
        reconstructed_B = self.generator_B.decode(latent_B_cross + noise_B_cross)

        # Losses
        generator_A_adverserial_loss = self.discriminator_A.calc_generator_loss(fake_A_cross)
        generator_B_adverserial_loss = self.discriminator_B.calc_generator_loss(fake_B_cross)

        generator_A_reconstruction_loss = self.mae_loss(fake_A_within, image_A)
        generator_B_reconstruction_loss = self.mae_loss(fake_B_within, image_B)

        generator_A_cyclic_loss = self.mae_loss(reconstructed_A, image_A)
        generator_B_cyclic_loss = self.mae_loss(reconstructed_B, image_B)

        kl_loss_A = torch.mean(torch.pow(latent_A, 2))
        kl_loss_B = torch.mean(torch.pow(latent_B, 2))

        kl_loss_cyclic_A = torch.mean(torch.pow(latent_A_cross, 2))
        kl_loss_cyclic_B = torch.mean(torch.pow(latent_B_cross, 2))

        total_loss = generator_A_adverserial_loss + generator_B_adverserial_loss \
             + 10 * generator_A_reconstruction_loss + 10 * generator_B_reconstruction_loss \
             + 10 * generator_A_cyclic_loss + 10 * generator_B_cyclic_loss \
             + 0.01 * kl_loss_A + 0.01 * kl_loss_B \
             + 0.01 * kl_loss_cyclic_A + 0.01 * kl_loss_cyclic_B

        self.generator_optimizer.zero_grad()
        total_loss.backward()
        self.generator_optimizer.step()

        # Write to tensorboard
        self.writer.add_scalar('Generator A loss', generator_A_adverserial_loss.item(), self.steps)
        self.writer.add_scalar('Generator B loss', generator_B_adverserial_loss.item(), self.steps)
        self.writer.add_scalar('Forward cycle loss', generator_A_cyclic_loss.item(), self.steps)
        self.writer.add_scalar('Reverse cycle loss', generator_B_cyclic_loss.item(), self.steps)

        self.writer.add_scalar('Reconstruction A loss', generator_A_reconstruction_loss.item(), self.steps)
        self.writer.add_scalar('Reconstruction B loss', generator_B_reconstruction_loss.item(), self.steps)

        self.writer.add_scalar('KL Loss A', kl_loss_A.item(), self.steps)
        self.writer.add_scalar('KL Loss B', kl_loss_B.item(), self.steps)

        self.writer.add_scalar('KL Loss A Cyclic', kl_loss_cyclic_A.item(), self.steps)
        self.writer.add_scalar('KL Loss B Cyclic', kl_loss_cyclic_B.item(), self.steps)


        print("Epoch : %d, Iteration : %d " % (epoch, iteration))


    def optimize_discriminator(self, image_A, image_B, epoch, iteration):
        self.steps += 1
        image_A = image_A.to(self.device)
        image_B = image_B.to(self.device)
        # Encoding
        latent_A, noise_A = self.generator_A.encode(image_A)
        latent_B, noise_B = self.generator_B.encode(image_B)

        # Decoding cross domain
        fake_A_cross = self.generator_A.decode(latent_B + noise_B)
        fake_B_cross = self.generator_B.decode(latent_A + noise_A)
        
        # Discriminator loss
        discriminator_A_loss = self.discriminator_A.calc_discriminator_loss(fake_A_cross.detach(), image_A)
        discriminator_B_loss = self.discriminator_B.calc_discriminator_loss(fake_B_cross.detach(), image_B)

        total_loss = discriminator_A_loss + discriminator_B_loss
        self.discriminator_optimizer.zero_grad()
        total_loss.backward()
        self.discriminator_optimizer.step()

        self.writer.add_scalar('Discriminator A loss', discriminator_A_loss.item(), self.steps)
        self.writer.add_scalar('Discriminator B loss', discriminator_B_loss.item(), self.steps)

    def update_optimizers(self):
        self.discriminator_scheduler.step()
        self.generator_scheduler.step()

    def save_network(self, epoch):
        """ Save all the networks
        """
        path = os.path.join(os.getcwd(), 'models', 'unit')

        torch.save(self.generator_A.state_dict(), '%s/generator_A_epoch_%d.pth' % (path, epoch))
        torch.save(self.generator_B.state_dict(), '%s/generator_B_epoch_%d.pth' % (path, epoch))
        torch.save(self.discriminator_A.state_dict(), '%s/discriminator_A_epoch_%d.pth' % (path, epoch))
        torch.save(self.discriminator_B.state_dict(), '%s/discriminator_B_epoch_%d.pth' % (path, epoch))
        torch.save(self.discriminator_optimizer.state_dict(), '%s/discriminator_optimizer_epoch_%d.pth' % (path, epoch))
        torch.save(self.generator_optimizer.state_dict(), '%s/generator_optimizer_epoch_%d.pth' % (path, epoch))
    
    def resume_training(self, epoch):
        path = os.path.join(os.getcwd(), 'models', 'unit')
        self.generator_A.load_state_dict(torch.load('%s/generator_A_epoch_%d.pth' % (path, epoch)))
        self.generator_B.load_state_dict(torch.load('%s/generator_B_epoch_%d.pth' % (path, epoch)))
        self.discriminator_A.load_state_dict(torch.load('%s/discriminator_A_epoch_%d.pth' % (path, epoch)))
        self.discriminator_B.load_state_dict(torch.load('%s/discriminator_B_epoch_%d.pth' % (path, epoch)))
        self.generator_optimizer.load_state_dict(torch.load('%s/generator_optimizer_epoch_%d.pth' % (path, epoch)))
        self.discriminator_optimizer.load_state_dict(torch.load('%s/discriminator_optimizer_epoch_%d.pth' % (path, epoch)))

        self.generator_scheduler = optim.lr_scheduler.StepLR(self.generator_optimizer, step_size = 100000, gamma = 0.5, last_epoch = epoch)
        self.discriminator_scheduler = optim.lr_scheduler.StepLR(self.discriminator_optimizer, step_size = 100000, gamma = 0.5, last_epoch = epoch)


    def train(self, start_epoch):
        if start_epoch > 0:
            self.resume_training(start_epoch)
            self.steps = start_epoch * len(self.dataset)
        for epoch in range(start_epoch + 1, 820):
            for i, data in enumerate(self.dataloader):
                imageA = data['A']
                imageB = data['B']

                self.optimize_discriminator(imageA, imageB, epoch, i)   
                self.optimize_generator(imageA, imageB, epoch, i)
                 
            # Update learning rate at the end of the epoch
            self.update_optimizers()

            imagesA, imagesB = self.dataset.load_test_images() # Load test images
            
            imagesA = imagesA.to(self.device)
            imagesB = imagesB.to(self.device)
            # Generate fake images
            fakesA, fakesB = self.forward(imagesA, imagesB)
            self.writer.add_image("Fake images A", vutils.make_grid(fakesA.data[:5]), epoch)
            self.writer.add_image("Real images A", vutils.make_grid(imagesA.data[:5]), epoch)
            self.writer.add_image("Fake images B", vutils.make_grid(fakesB.data[:5]), epoch)
            self.writer.add_image("Real images B", vutils.make_grid(imagesB.data[:5]), epoch)


            if epoch % 10 == 0:
                self.save_network(epoch)
    
    def test(self):
        transform = torchvision.transforms.ToPILImage()
        path = os.path.join(os.getcwd(), 'results', 'unit')
        for i, data in enumerate(self.dataloader):
            imageA = data['A']
            imageB = data['B']

            imageA = imageA.to(self.device)
            imageB = imageB.to(self.device)            
            fakeA, fakeB = self.forward(imageA, imageB)
            fakeA = transform(fakeA)
            fakeB = transform(fakeB)

            imageA = transform(imageA.cpu())
            imageB = transform(imageB.cpu())  

                      




    


if __name__ == '__main__':
    unit = UNITGAN()
    unit.train(start_epoch = 0)

    # unit = UNITGAN(training = False)
    
