'''
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
"""Convert a color image to grayscale and normalize the color range to [0,1]."""
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) = > x = x - u / std

'''

import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        
        hr_height, hr_width = hr_shape
        
        #1 Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        #2 glob.glob()directly retrieve paths recursively
        
        self.filesLR = sorted(glob.glob(root + '/LR'+"/*.*"))
        self.filesHR = sorted(glob.glob(root + '/HR'+"/*.*"))
        #print("type(self.filesLR): ", type(self.filesLR))

    def __getitem__(self, index):
        print ('Inside getitem(): index: ' +str(index))
        #print(index)
        #print(len(self.filesLR))
        #print(index % len(self.filesLR))
        
        # 1. read image from specific path
        #imgLR = Image.open(self.filesLR[index % len(self.filesLR)])
        #imgHR = Image.open(self.filesHR[index % len(self.filesHR)])

        imgLR = Image.open(self.filesLR[index])
        imgHR = Image.open(self.filesHR[index])

        #2. apply transformations
        img_lr = self.lr_transform(imgLR)
        img_hr = self.hr_transform(imgHR)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.filesLR)


dataloader = DataLoader( ImageDataset('testA', hr_shape=(32,32)), batch_size=4, shuffle=False)

#print("No of batches: ", len(dataloader))

for i, imgs in enumerate(dataloader):  # dataloader under the hood give next index value to __getItem__()
    
    imgs_lr = imgs["lr"]
    imgs_hr = imgs["hr"]
    print ('lr.size:', imgs_lr.shape)
    print ('hr.size:', imgs_hr.shape)
    print ('type:', type(imgs_lr))
    print ('type:', type(imgs_hr))
    break
    
