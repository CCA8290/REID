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
from model_2 import ft_net,PCB, PCB_test,ft_net_dense

#---------------------------------1、设置程序执行参数----------------------------------------
parser = argparse.ArgumentParser(description='Training')     #创建对象
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/content/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
opt = parser.parse_args()

config_path = os.path.join('/content/GPUID/model',opt.name,'opts.yaml') #载入配置文件
with open(config_path, 'r') as stream:
        config = yaml.load(stream,Loader=yaml.FullLoader)   #新版本需要加Loader=yaml.FullLoader
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.stride = config['stride']

if 'nclasses' in config: #在配置文件寻找分类数目
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')     #gpu id
name = opt.name                      #模型所在文件夹名称
test_dir = opt.test_dir              #测试集路径
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# 设置gpu
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

#---------------------------------2、载入数据集----------------------------------------------
#=====样本增强变换操作======
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
#=====载入测试样本=====
data_dir = test_dir
#这里由于不是训练，样本无需打乱顺序，win10仍然不支持多线程num_workers，需置零
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
#dataloaders 装入一个batch大小的数据
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes   #记录类名
use_gpu = torch.cuda.is_available()

#---------------------------------3、载入指定模型参数----------------------------------------
def load_network(network):
    save_path = os.path.join('/content/GPUID/model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

#---------------------------------4、借助预训练模型提取特征----------------------------------
#======实现图片的水平翻转=====
def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W，4维tensor，
    #按照列索引得到一个倒序序列，如（1,1,3,5）得到[4,3,2,1,0]
    #index_select按照倒序索引读取张量，就实现了图片水平翻转
    #第一个参数改为2可实现垂直翻转
    img_flip = img.index_select(3,inv_idx)
    return img_flip
#====特征提取，求距离=====
def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()  #batchx3x256x128 #PCB：384x192
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda() #32*512
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() #有6个parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            ff += outputs
        #求特征范数
        if opt.PCB:
            #特征尺寸(n,2048,6)
            # 为了让每一个parts受“公平对待”，对每一个2048维的零件特征都计算范数
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            #求范数，p表示幂指数(L2范数，欧式距离)，dim为指定维数求范数
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))        #expand_as(x):扩展张量szie同x，不共享内存；div为除法
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)   #[32,1]
            ff = ff.div(fnorm.expand_as(ff))        #根据tensor除法规则，不用expand也可以除

        features = torch.cat((features,ff.data.cpu()), 0)  #按照行拼接Tensor
    return features
#=======解析文件名=======
def get_id(img_path):
    camera_id = []      #相机id
    labels = []         #标签
    for path, v in img_path:    #imgs具体格式为列表套元组[(),()]
        filename = os.path.basename(path)   #返回路径最后的文件名，如ab/cd/ef.txt，返回ef.txt
        label = filename[0:4]               #前四位是标签
        camera = filename.split('c')[1]     #c后跟相机id
        if label[0:2]=='-1':                #-1表示图中的行人不属于测试集
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

#---------------------------------5、调用函数、核心程序--------------------------------------
gallery_path = image_datasets['gallery'].imgs       #.imgs获取图像路径和标签
query_path = image_datasets['query'].imgs
gallery_cam,gallery_label = get_id(gallery_path)    #获取相机id和标签
query_cam,query_label = get_id(query_path)
#加载训练好的模型
print('---------------测试---------------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)
if opt.PCB:
    model_structure = PCB(opt.nclasses)
model = load_network(model_structure)

#去掉最后的全连接分类层
if opt.PCB:
    model = PCB_test(model)
else:
    model.classifier.classifier = nn.Sequential()

#改为测试模式，固定BN和Dropout，防止样本通过上述层失真
#如果不加，由于存在BN层，即使不训练，权值也会改变。强调！！！
model = model.eval()
if use_gpu:
    model = model.cuda()

#=======提取特征========
with torch.no_grad():  #不计算梯度，不反向传播
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])
#======保存结果矩阵======
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,
          'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)   #保存结果矩阵
print(opt.name)   
result = '/content/GPUID/model/%s/result.txt'%opt.name
#tee  linux命令，输出内容到xxx同时在终端打印，将每次评测的结果记录起来
os.system('python /content/GPUID/evaluate_gpu.py | tee -a %s'%result)

