"""Objectives

    Image classification using pytorch
    Four key components of any ML system (in PyTorch):
      1)  Data (Images)
      2)  Model (CNN)
      3)  Loss (Cross Entropy)
      4)  Optimization (SGD, Adam, ..)
    Convolutional Neural Networks (CNNs)
    Overfit
    Data augmentation
    Transfer learning"""

import math
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Our libraries
from train import train_model
from model_utils import *
from predict_utils import *
from vis_utils import *

# some initial setup
np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(1234)

use_gpu
DATA_DIR = 'dataset'
sz = 224
batch_size = 16
os.listdir(DATA_DIR)

trn_dir = f'{DATA_DIR}/training_set'
val_dir = f'{DATA_DIR}/test_set'

os.listdir(trn_dir)

trn_fnames = glob.glob(f'{trn_dir}/*/*.jpg')
trn_fnames[:5]

img = plt.imread(trn_fnames[4])
plt.imshow(img);

train_ds = datasets.ImageFolder(trn_dir)
print(train_ds.transform)

#Datasets and Dataloaders in PyTorch

#    Dataset
#    A set of images.
#    Dataloader
#    Loads data from dataset behind the scene using concurrent threads.

train_ds = datasets.ImageFolder(trn_dir)
train_ds.classes
train_ds.class_to_idx
train_ds.root
train_ds.imgs

#Transformations
#Dataloader object uses these tranformations when loading data.

tfms = transforms.Compose([
    transforms.Resize((sz, sz)),  # PIL Image
    transforms.ToTensor(),        # Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # RGB

train_ds = datasets.ImageFolder(trn_dir, transform=tfms)
valid_ds = datasets.ImageFolder(val_dir, transform=tfms)

len(train_ds), len(valid_ds)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=8) #how many subprocesses to use for data
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=8)

inputs, targets = next(iter(train_dl))
out = torchvision.utils.make_grid(inputs, padding=3)
plt.figure(figsize=(16, 12))
imshow(out, title='Random images from training data')

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(56 * 56 * 32, 2) # 2 classes, dog and cat

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)            # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W)
        out = self.fc(out)
        return out

model = SimpleCNN()

# transfer model to GPU
if use_gpu:
    model = model.cuda()

print(model)

"""Loss function and optimizer"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

# Train
num_epochs = 10
losses = []

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dl):
        inputs = to_var(inputs)
        targets = to_var(targets)

        # forwad pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # loss
        loss = criterion(outputs, targets)
        losses += [loss.data]
        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        # report
        if (i + 1) % 50 == 0:
            print('Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_ds) // batch_size, loss.data))


#%%

plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Cross Entropy Loss');

# Accuracy on validation data

def evaluate_model(model, dataloader):
    model.eval()  # for batch normalization layers
    corrects = 0
    for inputs, targets in dataloader:
        inputs, targets = to_var(inputs, True), to_var(targets, True)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        corrects += (preds == targets.data).sum()

print('accuracy: {:.2f}'.format(100. * corrects / len(dataloader.dataset)))

evaluate_model(model, valid_dl)
evaluate_model(model, train_dl)
visualize_model(model, train_dl)
visualize_model(model, valid_dl)

plot_errors(model, valid_dl)

#Confusion matrix
y_pred, y_true = predict_class(model, valid_dl)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, train_ds.classes, normalize=True, figsize=(4, 4))

""" What is OVERFIT?

    The most important concept in ML!
    Simply, it means that your model is too complex for your problem.

    What we can do about it?

    Regularization
    Dropout
    Data Augmentation
    Transfer Learning
"""
# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((sz, sz)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Just normalization for validation
valid_transforms = transforms.Compose([
    transforms.Resize((sz, sz)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(f'{DATA_DIR}/training_set', train_transforms)
valid_ds = datasets.ImageFolder(f'{DATA_DIR}/test_set', valid_transforms)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

train_ds_sz = len(train_ds)
valid_ds_sz = len(valid_ds)

print('Train size: {}\nValid size: {} ({:.2f})'.format(train_ds_sz, valid_ds_sz, valid_ds_sz/(train_ds_sz + valid_ds_sz)))

class_names = train_ds.classes

inputs, targets = next(iter(train_dl))     # Get a batch of training data
out = torchvision.utils.make_grid(inputs)  # Make a grid from batch
plt.figure(figsize=(16., 12.))
imshow(out, title='Augmented Images');

#Look at the sizes of the images
fnames = glob.glob(f'{trn_dir}/*/*.jpg')
sizes = [Image.open(f).size for f in fnames]

hs, ws = list(zip(*sizes))

plt.figure(figsize=(12., 4.))
plt.hist(hs)
plt.hist(ws);


# create model
model = SimpleCNN()
if use_gpu:
    model = model.cuda()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

# train
model = train_model(model, train_dl, valid_dl, criterion, optimizer, num_epochs=5)

# load pre-trained ResNet18
model = load_pretrained_resnet50(model_path=None, num_classes=2)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

model = train_model(model, train_dl, valid_dl, criterion, optimizer, scheduler, num_epochs=2)
evaluate_model(model, valid_dl)
evaluate_model(model, valid_dl)
plot_errors(model, valid_dl)












