
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

def plot_loss_acc(timestamp, workspace, model_name, optimizer_name, epochs, train_losses, val_losses, train_accs, val_accs, img_size=[12,5]):
    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(img_size[0], img_size[1]))
    fig.suptitle(model_name + ' with ' + optimizer_name + ' optimizer')
    axs1.plot(epochs, train_losses, label='Training')
    axs1.plot(epochs, val_losses, label='Validation')
    axs1.set(xlabel='Epochs', ylabel='Loss')
    axs1.legend()
    axs2.plot(epochs, train_accs, label='Training')
    axs2.plot(epochs, val_accs, label='Validation')
    axs2.set(xlabel='Epochs', ylabel='Accuracy')
    axs2.legend()
    if os.path.isdir(os.path.join(workspace, 'graph')) != True:
      os.mkdir(os.path.join(workspace, 'graph'))
    fig.savefig(os.path.join(workspace, 'graph', timestamp + '.png'))   # save the figure to file
    # plt.show()

def plot_confusion_matrix(cm, classes, timestamp, workspace, model_name, cmap=plt.cm.Blues, save=True):
    plt.figure(figsize=(10,10))
    if os.path.isdir(os.path.join(workspace, 'graph')) != True and save == True:
      os.mkdir(os.path.join(workspace, 'graph'))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]) + '\n(' + format(round(cm[i,j]*100/207, 1)) + '%)', fontsize=20, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.title(model_name)
    if save:
        cm_path = timestamp + '_cm.png'
        plt.savefig(os.path.join(workspace, 'graph', cm_path))
    # plt.show()

def compute_AUC_scores(y_true, y_pred, labels):
    """
    Computes the Area Under the Curve (AUC) from prediction scores
    y_true.shape  = [n_samples, n_classes]
    y_preds.shape = [n_samples, n_classes]
    labels.shape  = [n_classes]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print('roc_auc_score for covid class: ', auc(y_true, y_pred))

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - DecisionTree')

    for i in range(5):
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_true, y_pred, pos_label=i)
        #print('roc_auc_score for DecisionTree: ', roc_auc_score(y_true, y_pred, multi_class='ovr'))
        plt.plot(false_positive_rate1, true_positive_rate1, label=labels[i])
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right', frameon=False)
    plt.show()

    # AUROC_avg = roc_auc_score(y_true, y_pred, multi_class='ovr')
    # print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
    # for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
    #     print('The AUROC of {0:} is {1:.4f}'.format(label, roc_auc_score(y, pred)))
    
