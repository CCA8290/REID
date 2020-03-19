# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torch.nn import init
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
'''
#训练之前可以运行一下该程序，查看模型结构
#import pretrainedmodels
#导入一些预训练模型如Resnet50，见下面的site  
#https://pypi.org/project/pretrainedmodels/

#何恺明初始化方法，针对ReLU的初始化，xavier仅在tanh中表现得好

#md=models.resnet50(pretrained=False)
#ad=map(id,md.layer1.parameters())
#print(*ad)

#input = Variable(torch.FloatTensor(32, 3, 256, 128))
#out=tst(input)
#print(out)
'''
x:([32, 2048, 6, 1])
part[i]:[32,2048]
predict[i][0]:[32,751]
predict[i][1]:[32,256]
'''
newlist=map(None,[1,2,3],[4,5,6])
print(next(newlist))
print(next(newlist))
print(next(newlist))