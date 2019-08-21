#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from load import process_image
from load import imshow
import time


# Set up model

# In[ ]:


def setup_model(structure = 'vgg16', dropout = 0.5, lr=0.001, power = 'GPU', hidden_layer = 512):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(p=dropout)),
        ('fc1', nn.Linear(num_features, hidden_layer)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer, 100)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(100,2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    if torch.cuda.is_available() and power == 'GPU':
        model.cuda()
    
    return model, criterion, optimizer


# train the model

# In[ ]:


def train_model(model, criterion, optimizer, train_dataloaders, vaild_dataloaders, power = 'GPU', epochs=1):
    accuracy_l = []
    training_l = []
    vaildation_l = []
    sensitivity_l = []
    specificity_l = []
    for e in range(epochs):
        running_loss = 0
        model.train()
        
        for images, labels in train_dataloaders:
            if torch.cuda.is_available() and power=='GPU':
                images,labels = images.to('cuda'), labels.to('cuda')
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            model.eval()
            accuracy = 0
            vaild_loss = 0
            
            accuracy = 0
            vaild_loss = 0
            
            TF_all = 0
            AF_all = 0
            TP_all = 0
            AP_all = 0
            AF = 0
            AP = 0
            
            with torch.no_grad():
                for images, labels in vaild_dataloaders:
                    if torch.cuda.is_available() and power=='GPU':
                        images,labels = images.to('cuda'), labels.to('cuda')
                    
                    
                    log_ps = model(images)
                    vaild_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    labels = labels.view(*top_class.shape)

                    equals = top_class == labels
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                    AF = sum (labels == 0).numpy()[0]
                    if AF != 0:
                        TF_all += sum(top_class[labels == 0] == 0).numpy()
                        AF_all += AF

                    AP = sum(labels == 1).numpy()[0]
                    if AP != 0:
                        TP_all += sum(top_class[labels == 1] == 1).numpy()
                        AP_all += AP
            
            
            sensitivity = (TP_all / AP_all)
            specificity = (TF_all / AF_all)

            accuracy = (TP_all + TF_all) / (AP_all + AF_all)

            running_loss = running_loss / len(train_dataloaders)
            training_l += [running_loss]
            valid_loss = vaild_loss / len(vaild_dataloaders)
            valid_loss = vaild_loss 
            vaildation_l += [valid_loss]

            accuracy_l += [accuracy]
            sensitivity_l += [sensitivity]
            specificity_l += [specificity]
            print("epoch {0}/{1} Training loss: {2:.4} ".format(e+1,epochs,running_loss),
                 "Vaildation loss: {0:.4}".format(valid_loss),
                 "Accurancy:{0:.4}".format(accuracy),
                 "Specificity:{0:.4}".format(specificity),
                 "Sensitivity:{0:.4}".format(sensitivity))
    else:
        print("Finished!")

