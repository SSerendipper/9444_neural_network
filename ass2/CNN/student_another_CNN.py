#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    brightness = (1, 8)
    contrast = (1, 8)
    saturation = (1, 8)
    hue = (0.3, 0.5)  
    if mode == 'train':
        return transforms.Compose([
          transforms.RandomCrop(75, padding=4),
          transforms.ColorJitter(brightness, contrast, saturation, hue),
          transforms.RandomRotation(30,fill=(0,0,255)),
          #have to resize on the CSE version of pytorch,
          #or it will make mistakes for different sizes of test and train
          transforms.Resize(80),
          transforms.ToTensor(),
          #converted to standard normal distribution, which makes the model easier to converge
          #mean and std are from https://pytorch.org/hub/pytorch_vision_densenet/
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif mode == 'test':
        return transforms.Compose([
          
          transforms.ToTensor(),
          #same as train
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
n_class = 8
batch_size = 25

#CNN
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 17 * 17, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adamax(net.parameters())
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data/data"
train_val_split = 0.8
epochs = 10


