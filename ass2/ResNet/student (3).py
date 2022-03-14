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

  Beginning with the simple CNN from Kuzu combined with the linear densenet from Frac,
after picking batchsize 100 from [20,30,100,200], applying image transformations, 
adding the hidden layers up to 5, adding the batch normalization and drop out,
picking Adam from [Adam,Adamax,Adadelta], the highest accuracy is only 55.38% at ep60.
  Then I move to classic networks. Both DenseNet and Inception v4 are over 50Mb, 
so I turn to ResNet. Though Res50 is over 90 Mb, after shifting the layer list from [3,4,6,3]
to [1,2,3,1], Res30 with 40 Mb just meets the requirements.
  With the help of Res_B and Res_C in paper[1], I slightly modificate the model structure, 
which could accelerate the convergence of the model while avoiding the loss of too much feature information.
  In the forum Ed, someones says the weight initialization will also improve the model. So
I use he initialization, which works well in Relu.
  Then I met the overfitting problem, as the old learning rate is no more proper now. After 
decreasing the learning rate to 0.015 and the StepLR size to 15, the model converges. And 
could get acc 54.62% at ep16. But the accuray still increases slowly since ep30.
  After experiment, I found simple structure will get higher accuracy, for [1,1,2,1] gets 68.5%,
[1,1,1,1] gets

  During the assignment 2, I nearly meet the bottelneck of colab GPU every day. Even I have
created "content/hw2" as tutor told, which just accelerates the time of first epoch but could not delay 
the comin gof bottelneck. It would be great if I installed pytorch locally at the beginning. 
This seriously affects me to try other models, such as WideResNet,ResNeXt and SENet, where 
both the performance and training speed could be improved.

paper[1]:{htong,zhiz,hzaws,zhongyue,junyuanx,mli}@amazon.com
Bag of Tricks for Image Classification with Convolutional Neural Networks,CVPR2019
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing

    1.ColorJitter->delete it could work better
    2.RandomGrayscale->works well on the result
    3.CenterCrop & RandomHorizontalFlip->mentioned in paper[1]
    4.Resize->have to resize on the CSE version of pytorch,or it will make mistakes for different sizes of test and train
    5.Normalize->supposed that it will make the model easier to converge, but it does not work well
    or maybe it should be applied on mode 'test'?
    """
    brightness = 0.1
    contrast = 0.1
    saturation = 0.1
    hue = 0.1 
    if mode == 'train':
        return transforms.Compose([
          transforms.RandomCrop(70),
          #transforms.CenterCrop(70),
          #transforms.ColorJitter(brightness, contrast, saturation, hue),
          #transforms.RandomRotation(30,fill=(0,0,255)),
          transforms.RandomRotation(45,interpolation =transforms.InterpolationMode.NEAREST),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomGrayscale(p=0.5),
          transforms.Resize(80),
          transforms.ToTensor(),
          #mean and std are from https://pytorch.org/hub/pytorch_vision_densenet/
          #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif mode == 'test':
        return transforms.Compose([
          
          transforms.ToTensor(),
          #same as train
          #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
"""
The basic ResNet50 is from 
https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
Res50 after trainging 60 epoch will be over 90 MB, then I try to use use fewer layers
and imply some modifications based on paper[1]:
"""

"""
-ResNet-C
As "A 7 × 7 convolution is 5.4 times more expensive than a 3 × 3 convolution."
Replace a 7 * 7 to three 3 * 3
The reuslt torch.size will be [100,64,28,23] instead of [100,64,20,20].
"""
def Conv_ResNet_C(in_planes, places):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=32,kernel_size=3,stride=2,padding=(4,2), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=(4,2), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=(4,2), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
"""
-ResNet-B
ResNet used to ignore three-quarters of the input feature map for the stride is too large,
reduce it to get more features.
For kernel_size > stride, the convolution kernel can traverse all the 
information on the input characteristic graph in the process of moving.
"""
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #The mechanism of BN layer may be to smooth the distribution of hidden layer input,
        #help the random gradient descent, and alleviate the negative impact of random gradient
        # descent weight update on subsequent layers.
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.relu(self.bn2(self.conv2(x)))
        
        x = self.bn3(self.conv3(x))
        
        #downsample since make_kayer 2
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Network(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(Network, self).__init__()
        self.in_channels = 64
        
        self.conv1 = Conv_ResNet_C(in_planes = 3, places= 64)

        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #why no softmax:
        #1.the order preserving property of softmax
        #2.reduce the amount of calculation
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                #nn.AvgPool2d(2, stride=2),
                #bugs:RuntimeError: The size of tensor a (23) must match the size of tensor b (11) at non-singleton dimension 3
                #do not know how to solve
                #could not preform ResNet_D
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
"""
[2,2,3,2] convenges faster than [2,2,4,2], but over 50Mb
[1,2,3,2] works even better than [2,2,3,2], get 25.72% at epoch 1,but still over 50 Mb
[1,2,2,1] 36.24Mb, if use it, the step_size of scheduler should be 20
[1,2,3,1] 40.52Mb->adding a randomcrop 40.57Mb
  At ep 10, there are more than 110 images classified correctly for type 0/5/6/7, while there
is a severe mismatch between [1,6,7],[2,5],[3,5,6] and [4,1,2].
  At ep 40, the accuracy got stuck in around 57%, even lower than 59.75% at ep28. There are more than 110 
images classified correctly for type 0/3/4/5/7, while there is a severe mismatch between [1,5],[2,5],and [6,5].
0.015+StepLR_size = 15->57.56% ep64
simplify the structure
[1,1,2,1]+ Grayscale p=0.5 
lr = 0.003 35.16Mb acc = 55.19% at ep7
lr = 0.0005 acc = 64.06% at ep16
lr = 0.0001 acc = 55% at ep20 stuck
lr = 0.001 acc = 64.69% at ep16
"""
net = Network(Bottleneck, [1,1,1,1], 8, 3)
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
"""
pick 0.05 from lr = [0.001,0.005,0.05,0.1],but mets overfitting on final ResNet
0.03->60% acc at ep30 but get tracked,need to decrease both learning rate and StepLR_size
"""
optimizer = optim.Adam(net.parameters(), lr=0.0007)
#for classification
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
"""
he initialization， works better than Xavier Initialization in Relu,
based on https://blog.csdn.net/qq_39938666/article/details/88374110
"""    
def weights_init(m):
    return
"""
Adjust the learning rate according to the epoch training times
to prevent falling into local optimization.
"""
scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) 

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 100
#training more than 50
epochs = 85
