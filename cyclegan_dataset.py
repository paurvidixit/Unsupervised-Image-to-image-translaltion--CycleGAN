import os 
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob
import random
import torchvision.transforms as transforms
import torch
import numpy as np

class CycleGANTestDataset(Dataset):
    def __init__(self, base_dir, folder = 'A'):
        self.base_dir = base_dir
        path = os.path.join(base_dir, 'test' + folder)
        self.images = glob.glob(path + '/*.jpg')

        self.size = len(self.images)
        self.transform = self.transform_image()

    def __len__(self):
        return self.size

    def transform_image(self):
        """ Get the transformations to be applied on the images
        """
        transform_list = []
        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transforms = transform_list) # Compose all the transforms together and return

    def __getitem__(self, index):
        index = index % self.size
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transform(image)

        return {'path' : self.images[index], 'data' : image}

class CycleGANDataset(Dataset):
    def __init__(self, base_dir, phase = 'train'):

        self.base_dir = base_dir
        A_path = os.path.join(base_dir, phase + 'A')
        B_path = os.path.join(base_dir, phase + 'B')

        # Get images for A and B
        self.A_images = glob.glob(A_path + '/*.jpg')
        self.B_images = glob.glob(B_path + '/*.jpg')
        # Get length of dataset for A and B
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)

        self.transform = self.transform_image()

    def __len__(self):
        return max(self.A_size, self.B_size) # Length of the dataset is the maximum of the lengths of A and B datasets

    def __getitem__(self, index):
        # Get indices for A, B. Since dataset is unaligned, imageB index is random
        A_index = index % self.A_size
        B_index = random.randint(0, self.B_size - 1)

        # Load the images
        imageA = Image.open(self.A_images[A_index]).convert('RGB')
        imageB = Image.open(self.B_images[B_index]).convert('RGB')

        # Apply transforms on the images
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)

        # Return dictionary containing the images
        return {'A' : imageA, 'B' : imageB}


    def transform_image(self):
        """ Get the transformations to be applied on the images
        """
        transform_list = []
        # Random horizontal flip
        transform_list.append(transforms.RandomHorizontalFlip(p = 0.5))

        # Convert to tensor and Normalize
        transform_list.append(transforms.ToTensor())
        # transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms = transform_list) # Compose all the transforms together and return



    def load_test_images(self):
        A_path = os.path.join(self.base_dir, 'testA')
        B_path = os.path.join(self.base_dir, 'testB')

        # Get images for A and B
        A_images = glob.glob(A_path + '/*.jpg')
        B_images = glob.glob(B_path + '/*.jpg')
        transform = transforms.ToTensor()
        
        imagesA = []
        imagesB = []
        n_A = len(A_images) # Length of dataset
        indices_A = np.random.choice(n_A, size = 1, replace = False) # Choose 1 image randomly from test set

        n_B = len(B_images) # Length of dataset
        indices_B = np.random.choice(n_B, size = 1, replace = False) # Choose 1 image randomly from test set


        A_images = [A_images[index] for index in indices_A]
        B_images = [B_images[index] for index in indices_B]

        for index in range(len(A_images)): 
            imageA = Image.open(A_images[index]).convert('RGB')
            imageB = Image.open(B_images[index]).convert('RGB')
            imageA = transform(imageA)
            imageB = transform(imageB)

            imageA = torch.unsqueeze(imageA.data, 0) # Expand dimensions of image to include batch size dimension (Dimension 0)
            imageB = torch.unsqueeze(imageB.data, 0) 

            imagesA.append(imageA)
            imagesB.append(imageB)
        
        imagesA = torch.cat(imagesA, dim = 0) # Concatenate the images along batch_size dimension to form a tensor
        imagesB = torch.cat(imagesB, dim = 0)
    
        return imagesA, imagesB



# actual_path = '/home/daniel/DL/Project/leftImg8bit_trainvaltest/leftImg8bit/train/*/'
# actual_images = glob.glob(actual_path + '*.png')
# actual_images = sorted(actual_images)
# label_path = '/home/daniel/DL/Project/gtFine_trainvaltest/gtFine/train/*/'
# label_images = glob.glob(label_path + '*_color.png')
# label_images = sorted(label_images)

# actual_converted_path = '/home/daniel/DL/Project/train/actual/'
# label_converted_path = '/home/daniel/DL/Project/train/label/'

# counter = 0
# for label_image in label_images:
#     image = cv2.imread(label_image)
#     image = cv2.resize(image, (128, 128), interpolation = cv2.INTER_LINEAR)
#     path = label_converted_path + str(counter) + '.png'
#     cv2.imwrite(path, image)
#     counter += 1

# counter = 0
# for actual_image in actual_images:
#     image = cv2.imread(actual_image)
#     image = cv2.resize(image, (128, 128), interpolation = cv2.INTER_LINEAR)
#     path = actual_converted_path + str(counter) + '.png'
#     cv2.imwrite(path, image)
#     counter += 1



