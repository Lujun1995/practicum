#!/usr/bin/env python
# coding: utf-8

# load in the data

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

def loaddata(data_dir = "/Users/pro/Desktop/practicum/data/new_data/Case/cancer", batch = 32):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/val'
    test_dir = data_dir + '/test'
    
    # For training images, we resize in to 300 *300, 
    # random rotation 30 and random horizontal flip to avoid overfitting the training
    # and crop into the size 224*224
    # we use transform learning so we nomarlize the training images as they do
    train_transforms = transforms.Compose([transforms.Resize(300),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                          ])
    
    
    # For vaidation and test images, we only resize, crop and nomarlize.
    vaild_transforms = transforms.Compose([transforms.Resize(300),
                                           transforms.CenterCrop((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                          ])
    
    test_transforms = transforms.Compose([transforms.Resize(300),
                                           transforms.CenterCrop((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                          ])  
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    vaild_data = datasets.ImageFolder(valid_dir, transform=vaild_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_data, batch_size=batch)
    vaild_dataloaders = torch.utils.data.DataLoader(vaild_data, batch_size=batch)
    return train_dataloaders, vaild_dataloaders, test_dataloaders


# In[ ]:



def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(300),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    img_tensor = transform(img_pil)
    return img_tensor


# In[ ]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
    transform = transforms.Compose([transforms.Resize(300),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    img_tensor = transform(img_pil)
    return img_tensor


#function to show image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

#show multiple images
def multishow(images, labels, num = 3):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    fig, ax = plt.subplots(2, num, figsize=(10,8))
    fig.suptitle('Transformed scans of histopathology images',fontsize=20)
    for i in range(num):
        image_postive = images[labels == 1]
        image = image_postive[i].numpy().transpose((1, 2, 0))
        image = image * std + mean
        image = np.clip(image, 0, 1)
        ax[0,i].imshow(image)
    ax[0,0].set_ylabel('positive samples', size='large')

    for i in range(num):
        image_negative= images[labels == 0]
        image = image_negative[i].numpy().transpose((1, 2, 0))
        image = image * std + mean
        image = np.clip(image, 0, 1)
        ax[1,i].imshow(image)
    ax[1,0].set_ylabel('negative samples', size='large')

