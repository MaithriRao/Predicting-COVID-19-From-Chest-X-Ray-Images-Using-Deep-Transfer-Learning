


import os
import datetime
import json
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from time import sleep
import datetime
import cv2
import shutil
from tqdm import tqdm


class TwoInputsNet(nn.Module):

  def __init__(self, model1, model2, num_classes):

    super(TwoInputsNet, self).__init__()

    # self.model1 = models.densenet161(pretrained=True)
    # self.model1.classifier = nn.Sequential(
    # nn.Dropout(0.3),
    # nn.Linear(2208, 1024)
    # )

    # self.model2 = models.inception_v3(pretrained=True)
    # self.model2.fc = nn.Linear(2048, 1024)

    self.model1 = model1
    self.model2 = model2
    self.fc2 = nn.Linear(2048, num_classes)

  def forward(self, input1, input2):

    c = self.model1(input1)
    f = self.model2(input2)

    combined = torch.cat((c,f), dim=1)

    out = self.fc2(F.relu(combined))

    return out

def list_toTorch(list):
    return torch.from_numpy(np.array(list))
    
def load_pretrained_model(model_name):
    if model_name == 'ResNeXt50':
        model = models.resnext50_32x4d(pretrained=True)
    elif model_name == 'Inception_v3':
        model = models.inception_v3(pretrained=True, aux_logits=False)
    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=True)
    return model

def edit_model(model_name, model, dropout, prob, freeze, num_classes):
    if freeze:
        print ("\n[INFO] Freezing feature layers...")
        for param in model.parameters():
            param.requires_grade=False
        sleep(0.5)
        print("-"*50)
    if model_name == 'DenseNet161':
        num_ftrs = model.classifier.in_features
        if dropout:
            model.classifier = nn.Sequential(
                nn.Dropout(prob),
                nn.Linear(num_ftrs, num_classes))
        else:
            model.classifier = nn.linea(num_ftrs, num_classes)
    elif model_name == 'Inception_v3':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        if dropout:
            model.fc = nn.Sequential(
                nn.Dropout(prob),
                nn.Linear(num_ftrs, num_classes))
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_transform(model_name):
    if model_name == 'Inception_v3':
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def create_model(workspace, dataset, num_classes, model_name, freeze, dropout, prob, lr, momentum, step_size, gamma, criterion_name, optimizer_name, device):
    if model_name == 'ResNeXt50' or model_name =='Inception_v3' or model_name == 'DenseNet161':
        model = load_pretrained_model(model_name)
        model = edit_model(model_name, model, dropout, prob, freeze, num_classes)
        transform = create_transform(model_name)
    elif len(model_name) == 2:
        transform = []
        model1 = load_pretrained_model(model_name[0])
        model1 = edit_model(model_name[0], model1, dropout, prob, freeze, 1024)
        transform.append(create_transform(model_name[0]))
        model2 = load_pretrained_model(model_name[1])
        model2 = edit_model(model_name[1], model2, dropout, prob, freeze, 1024)
        transform.append(create_transform(model_name[1]))
        model = TwoInputsNet(model1, model2, num_classes)


    if optimizer_name == 'Adam':
        optimizer_conv = optim.Adam(model.parameters(), lr=lr)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

    if criterion_name == 'Cross Entropy':
        criterion = nn.CrossEntropyLoss()


    return model, optimizer_conv, criterion, exp_lr_scheduler, transform
