# Objective

To perform unsupervised image to image translation
a Generative adversarial network converts images from an input
domain to target domain.

Image to image translation is the task of taking input images
belonging to a certain domain and producing corresponding
output images in the target domain. In recent years GANs
first introduced by Goodfellow have primarily
been used to perform this task. Image generation tasks using
GANs was first introduced by Radford. There are
two primary approaches to this task i.e. unpaired image to
image translation and paired image to image translation. In
paired image to image translation the network is fed pairs of
corresponding input-output images. However, it is extremely
difficult and in some cases impossible to get pairs of corresponding
input output image pairs. For example for the task of
translating images from summer to winter, getting the same
image in both summer and winter would be a difficult data
collcetion process.


# Implementation
For this project, we used the Yosemite summer and winter
dataset, which contains around 2500 images from Yosemite
taken in winter and summer with a rough 50-50 split. Around
2100 images were used for training and the rest used for test.
All images were of size 256*256. Since there is no quantitative metric to quantify the performance of the network, we omitted
the validation set.

We also tried out a few changes to
the hyperparameters of the original CycleGAN architecture to
obtain different and possibly better results. All our code was
implemented using the PyTorch framework and trained on a
RTX 2060. Each epoch of the CycleGAN took roughly 10-
15 minutes to train while the UNIT took around 30 minutes
per epoch.

# Results

![Results](/docs/gan.PNG)