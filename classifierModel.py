import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import torch.nn.functional as F


state_dict=torch.load("/home/mr-vaani/Downloads/DIGIT-CLASSIFIER-PARAMS.pth",weights_only=True,map_location="cpu")


class DIGITCLASSIFIER(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(1,8,3,1,1)
    self.conv2=nn.Conv2d(8,16,3,1,1)
    self.conv3=nn.Conv2d(16,32,3,1,1)
    self.conv4=nn.Conv2d(32,64,3,1,1)
    self.pool=nn.MaxPool2d(2,2)
    self.fc1=nn.Linear(64*14*14,128)
    self.fc2=nn.Linear(128,10)

  def forward (self,x):
    x=F.relu(self.conv1(x))
    x=F.relu(self.conv2(x))
    x=F.relu(self.conv3(x))
    x=self.pool(F.relu(self.conv4(x)))
    x=x.view(x.size(0),-1)
    x=F.relu(self.fc1(x))
    x=self.fc2(x)
    return x
  
