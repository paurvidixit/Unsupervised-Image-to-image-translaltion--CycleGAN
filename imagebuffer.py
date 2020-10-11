import torch
import torch.nn as nn
import numpy as np

class ImageBuffer:
    def __init__(self, buffer_size):
        """ Buffer to store the images generated from generator to train discriminator on history of images to improve the stability of CycleGAN.
        As used in the paper.
        """

        self.buffer_size = buffer_size
        self.cur_size = 0
        self.buffer = []
    
    def add_and_get_image(self, images):
        """ Returns images from the buffer and also adds the input images to the buffer
        If buffer is not full, only latest images are returned and also added to the buffer

        If buffer is fulled, earlier images are returned with probability 50% and current images replace them in the buffer
        """
        buffer_images = []

        for image in images:
            image = torch.unsqueeze(image.data, 0) # Expand dimensions of image to include batch size dimension (Dimension 0)
            if self.cur_size < self.buffer_size: # Buffer is not full
                self.cur_size += 1
                self.buffer.append(image)
                buffer_images.append(image)

            else: # Buffer is full
                p = np.random.uniform(low = 0.0, high = 1.0)
                if p < 0.5: # 50% probability of returning image from the buffer
                    index = np.random.randint(low = 0, high = self.buffer_size)
                    
                    buffer_image = self.buffer[index].clone()
                    buffer_images.append(buffer_image)
                    self.buffer[index] = image
                else: # 50% probability of returning the new image
                    buffer_images.append(image)
        buffer_images = torch.cat(buffer_images, dim = 0) # Concatenate the images along batch_size dimension to form a tensor
        return buffer_images
