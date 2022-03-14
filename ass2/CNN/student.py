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
batch_size = 100
l_input = 384
#CNN
class Network(nn.Module):

    def __init__(self):
        super(Network,self).__init__()
        #CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=5,stride=1,padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = torch.nn.Conv2d(in_channels=50, out_channels=64,kernel_size=5,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d((2,4), stride=5)
        self.full_1 = nn.Linear(l_input,batch_size)
        self.full_2 = nn.Linear(batch_size,n_class)
  


        

        
    def forward(self, input):
      
      hid_1 = (F.relu(self.bn1(self.maxpool(self.conv1(input)))))
      hid_2 = (F.relu(self.bn2(self.maxpool(self.conv2(hid_1)))))
      hid_2 = hid_2.view(-1,l_input)
      hid_3 = F.relu(self.full_1(hid_2))
      x_output=F.log_softmax(self.full_2(hid_3),dim = 1)
      return x_output

      





net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
learning_rate = 0.001
optimizer = optim.AdamW(net.parameters(), lr = learning_rate,weight_decay=0.0001)


loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return
epochs = 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.85


