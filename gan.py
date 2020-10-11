import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.utils as vutils
import torchvision.transforms as transforms

from generator import Generator
from discriminator import Discriminator

class GAN:
    def __init__(self):
        self.device = torch.device('cuda')

        self.generator = Generator(100, 1, 64).to(self.device)
        self.discriminator = Discriminator(1, 64).to(self.device)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        print(self.generator)
        print(self.discriminator)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))

        self.loss_criterion = nn.BCELoss()

        train_dir = os.path.join(os.getcwd(), 'train', 'Imagenet')
        test_dir = os.path.join(os.getcwd(), 'test')
        # self.dataset = datasets.MNIST(train_dir, download=True,
        #                    transform=transforms.Compose([
        #                        transforms.Resize(64),
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.5,), (0.5,)),
        #                    ]))

        self.dataset = datasets.ImageFolder(root = train_dir, transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        self.data_loader = DataLoader(self.dataset, batch_size = 128, shuffle = True, num_workers = 2)
        self.writer = SummaryWriter(comment = 'dcgan_imagenet')

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, mean = 0.0, std = 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, mean = 1.0, std = 0.02)
            nn.init.constant_(m.bias.data, val = 0.0)
    
    def train(self):
        fixed_noise = torch.randn(size = (128, 100, 1, 1), device = self.device)
        real_label = 1
        fake_label = 0
        for epoch in range(100):
            real_losses = []
            fake_losses = []
            gen_losses = []
            dis_losses = []
            for i, data in enumerate(self.data_loader, 0):
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()

                
                real_images = data[0].to(self.device)
                batch_size = real_images.size(0)

                noise = torch.randn(size = (batch_size, 100, 1, 1), device = self.device)
                fake_images = self.generator(noise)

                real_labels = torch.full(size = (batch_size,), fill_value = real_label, device = self.device)
                fake_labels = torch.full(size = (batch_size,), fill_value = fake_label, device = self.device)

                real_output = self.discriminator(real_images)
                fake_output = self.discriminator(fake_images.detach())

                real_loss = self.loss_criterion(real_output, real_labels)
                fake_loss = self.loss_criterion(fake_output, fake_labels)
                real_loss.backward()
                fake_loss.backward()

                self.dis_optimizer.step()
                dis_loss = real_loss + fake_loss
                real_losses.append(real_loss.item())
                fake_losses.append(fake_loss.item())
                dis_losses.append(dis_loss.item())

                fake_output = self.discriminator(fake_images)
                gen_loss = self.loss_criterion(fake_output, real_labels)
                gen_loss.backward()
                self.gen_optimizer.step()
                gen_losses.append(gen_loss.item())

                print("Epoch : %d, Iteration : %d, Generator loss : %0.3f, Discriminator loss : %0.3f, Real label loss : %0.3f, Fake label loss : %0.3f"
                                % (epoch, i, gen_loss.item(), dis_loss.item(), real_loss.item(), fake_loss.item()))

            real_loss = np.mean(np.array(real_losses))
            fake_loss = np.mean(np.array(fake_losses))
            dis_loss = np.mean(np.array(dis_losses))
            gen_loss = np.mean(np.array(gen_losses))

            fake_images = self.generator(fixed_noise).detach()
            self.writer.add_image("images", vutils.make_grid(fake_images.data[:128]), epoch)

            self.writer.add_scalar('Real Loss', real_loss, epoch)
            self.writer.add_scalar('Fake Loss', fake_loss, epoch)
            self.writer.add_scalar('Discriminator Loss', dis_loss, epoch)
            self.writer.add_scalar('Generator Loss', gen_loss, epoch)

            self.save_network(epoch)

    def save_network(self, epoch):
        path = os.path.join(os.getcwd(), 'models')
        torch.save(self.generator.state_dict(), '%s/generator_epoch_%d.pth' % (path, epoch))
        torch.save(self.discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (path, epoch))
        
if __name__ == "__main__":
    gan = GAN()
    gan.train()