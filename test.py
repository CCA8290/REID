# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms

import os
import scipy.io
import yaml
import math

'''
#实现图片水平翻转
a = torch.linspace(1, 15, steps=15).view(1,1,3, 5)
print(a)
b = torch.index_select(a, 3, torch.tensor([4,3,2,1,0]))
print(b)
t=torch.arange(5,-1,-1)
print(t)

inv_idx = torch.arange(a.size(3)-1,-1,-1).long()
print(torch.arange(a.size(3)-1,-1,-1).long())
img_flip = a.index_select(3,inv_idx)
print(img_flip)

net=models.resnet50(pretrained=True)
net.add_module('test',nn.Sequential(
    nn.Linear(3,5),
    nn.ReLU(),
    nn.Linear(5,7),
    nn.ReLU(),
    nn.Linear(7,3)
))
net.add_module('test2',nn.Sequential())
print(net)

a=torch.tensor([[1,2],[3,4]])
b=torch.tensor([[1,2,3],[4,5,6]])
#print(torch.mm(a,b))
print(b.view(-1,1))

a=np.array([3,2,2,-1,7,6])
b=np.array([3,2,1])
c=np.intersect1d(a,b,assume_unique=True)
print(c)
'''
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainwindow import *
class MyWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyWindow,self).__init__(parent)
        self.setupUi(self)

if __name__=='__main__':
    app=QApplication(sys.argv)
    myWin=MyWindow()
    myWin.show()
    sys.exit(app.exec_())