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

def test_model(model, test_dataloaders, criterion, power = 'GPU'):
    size = 0
    correct = 0
    test_loss = 0
    TF_all = 0
    AF_all = 0 
    TP_all = 0
    AP_all = 0
    AF = 0
    AP = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloaders:
            if torch.cuda.is_available() and power=='GPU':
                images,labels = images.to('cuda'), labels.to('cuda')
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            log_ps = model(images)
            labels = labels.view(*top_class.shape)

            equals = top_class == labels
            AF = sum (labels == 0).numpy()[0]
            if AF != 0:
                TF_all += sum(top_class[labels == 0] == 0).numpy()
                AF_all += AF

            AP = sum(labels == 1).numpy()[0]
            if AP != 0:
                TP_all += sum(top_class[labels == 1] == 1).numpy()
                AP_all += AP
            size += labels.size(0)
            correct += equals.sum().item()
    test_accuracy = correct / size    
    sensitivity = (TP_all / AP_all)
    specificity = (TF_all / AF_all)

    print("Accurancy:{:.4f}".format(test_accuracy))
    print(sensitivity)
    print(specificity)

