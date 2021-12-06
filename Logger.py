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
#import adabound
from matplotlib import pyplot as plt
from time import sleep
import datetime
import cv2
import shutil
from tqdm import tqdm


def create_train_log(workspace, train_accs, train_losses, train_f1_list, val_accs, val_losses, val_f1_list, model_name, optimizer_name, criterion_name, lr, momentum, step_size, gamma, num_epochs):
    save_path = os.path.join(workspace, "log")
    timestamp = str(datetime.datetime.now()).split('.')[0]
    log = json.dumps({
        'model': model_name,
        'optimizer': optimizer_name,
        'loss function': criterion_name,
        'timestamp': timestamp,
        'num_epoch': num_epochs,
        'lr': lr,
        'momentum': momentum,
        'step_size': step_size,
        'gamma': gamma,
        'last_train_acc': float('%.5f' % train_accs[-1]),
        'best_train_acc': float('%.5f' % max(train_accs)),
        'last_train_loss': float('%.5f' % train_losses[-1]),
        'best_train_loss': float('%.5f' % min(train_losses)),
        'last_train_f1': float('%.5f' % train_f1_list[-1]),
        'best_train_f1': float('%.5f' % max(train_f1_list)),
        'last_val_acc': float('%.5f' % val_accs[-1]),
        'best_val_acc': float('%.5f' % max(val_accs)),
        'last_val_loss': float('%.5f' % val_losses[-1]),
        'best_val_loss': float('%.5f' % min(val_losses)),
        'last_val_f1': float('%.5f' % val_f1_list[-1]),
        'best_val_f1': float('%.5f' % max(val_f1_list)),
        'train_accuracies': train_accs,
        'train_losses': train_losses,
        'train_f1_list': train_f1_list,
        'val_accuracies': val_accs,
        'val_losses': val_losses,
        'val_f1_list': val_f1_list
    }, ensure_ascii=False, indent=4)
    save_log(log, save_path)

def create_test_log(workspace, cm, test_acc, f1_test, model_name):
    save_path = os.path.join(workspace, "log")
    timestamp = str(datetime.datetime.now()).split('.')[0]
    log = json.dumps({
        'model': model_name,
        'timestamp': timestamp,
        'test_acc': float('%.5f' % test_acc),
        'test_f1': float('%.5f' % f1_test),
        'confusion matrix': cm.tolist()
    }, ensure_ascii=False, indent=4)
    save_log(log, save_path, train=False)

def save_log(log, save_path, train=True):
    timestamp = json.loads(log)['timestamp']
    if train:
        log_name = timestamp.split(' ')[0] + '.log'
    else:
        log_name = timestamp.split(' ')[0] + '_TEST.log'
    if os.path.isdir(save_path) != True:
        os.mkdir(save_path)
    log_file = os.path.join(save_path, log_name)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write("{}\n".format(log))
