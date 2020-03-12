# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg') #Linux系统在没有GUI的情况下可以绘图
import matplotlib.pyplot as plt
import time
import os
from model_2 import ft_net,PCB #从model导入定义的模型
import yaml
from shutil import copyfile

#---------------------------------1、设置程序执行参数----------------------------------------
#参数设置
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='/content/Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride') #Resnet里靠近输出的layer4层conv2和下采样步长
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
opt = parser.parse_args()   #将参数存到opt

data_dir = opt.data_dir #已分类数据集目录
name = opt.name    #模型存储的名称
str_ids = opt.gpu_ids.split(',') #显卡序号按照逗号分割存入列表
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)   #转为int
    if gid >=0:
        gpu_ids.append(gid)   #显卡序号存入整数数字列表

#设置gpu
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True #cudnn自动寻找适合当前配置的高效算法

#---------------------------------2、载入数据并增强样本--------------------------------------
transform_train_list = [
        transforms.Resize((256,128),interpolation=3), #调整图像尺寸为256x128，插值方法选Image.BICUBIC
        transforms.Pad(10),  #上下左右填充10个像素
        transforms.RandomCrop((256,128)), #随机裁剪
        transforms.RandomHorizontalFlip(),#随机水平翻转。默认概率0.5
        transforms.ToTensor(),#转为tensor，归一化至[0,1]
        #对数据按通道进行标准化，即先减均值，再除以标准差，注意顺序是c，h，w
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
#-------------如果使用PCB--------------
if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),#如果用PCB，尺寸变为384x192
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
print(transform_train_list)  #输出针对训练集的transform操作
data_transforms = {
    'train': transforms.Compose( transform_train_list ),  #串联列表中的多个transform操作
    'val': transforms.Compose(transform_val_list),
}
#读取训练集
train_all = ''
if opt.train_all:
     train_all = '_all'   #定义参数，判定训练训练集还是训练集+验证集

image_datasets = {}
#ImageFolder：文件按文件夹分类放好，文件夹名为某一类的名字，该方法能按类读取数据
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])  #读取train或者train_all
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])    #读取val
#DataLoader：将dataset根据batch size大小等参数装入一个batch Size大小的Tensor，用于后续训练。
#num_workers是指用8个线程，win10不支持多线程，改为0
#pin_memory如果True，数据加载器会在返回之前将Tensors复制到CUDA固定内存
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True)
                                             for x in ['train', 'val']}
#记录训练集和验证集的大小（图片数量），Market输出{'train': 12936, 'val': 751}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes   #记录类名['0002', '0007',...]
use_gpu = torch.cuda.is_available()  #检查是否有可用gpu资源（或者说是否有cuda支持）

since = time.time()    #从这里开始记录时间
inputs, classes = next(iter(dataloaders['train']))  #迭代读取一个batch的数据，包括训练图片和对应的类
print(time.time()-since)  #记录一下时间

#---------------------------------3、训练模型核心函数----------------------------------------
#要实现的功能包括：学习率调整、模型保存、记录loss和error
#scheduler是调整学习率参数，来自于torch.optim.lr_scheduler
y_loss = {} # 用来记录loss
y_loss['train'] = []
y_loss['val'] = []
y_err = {} #用来记录error
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  #计时
    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)         #输出训练到第几轮
        #每一轮都包含训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()   #调整学习率
                model.train(True)  #模型训练模式
            else:
                model.train(False)  #验证模式

            running_loss = 0.0
            running_corrects = 0.0
            #遍历数据
            for data in dataloaders[phase]:
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: #最后一个batch不训练
                    continue
                #print(inputs.shape)
                #装在Variable里
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach()) #detach方法，切断反向传播，使inputs不参与参数更新
                    labels = Variable(labels.cuda().detach()) #加上cuda能够用gpu加速
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()       #将梯度置0以便后面计算新的梯度
                #前向传播
                if phase == 'val':
                    with torch.no_grad():  #如果是验证集，则不需要算梯度，不需要反向传播
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)   #shape：对于market是batch*751

                #根据有没有用到PCB
                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)  #返回每列最大值及索引得到一个batch的分类结果
                    loss = criterion(outputs, labels)   #求出loss
                else:
                    part = {}
                    sm = nn.Softmax(dim=1) #对单个样本输出结果softmax，按照每一行
                    num_part = 6  #切成6块
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)   #分类结果

                    loss = criterion(part[0], labels)  #每个parts单独计算损失
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)  #加和每一个part的loss

                #对于训练样本反向传播、更新权值
                if phase == 'train':
                    loss.backward()   #反向传播求梯度
                    optimizer.step()  #更新权值

                #统计损失和误差
                running_loss += loss.item() * now_batch_size   #loss.item代替了老版本的loss.data[0]
                running_corrects += float(torch.sum(preds == labels.data))  #统计预测和实际相等的数量
            #统计每一轮的loss和error
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #记录loss和error
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            #模型保存
            if phase == 'val':  #训练集跑完验证完毕才保存参数、绘图
                last_model_wts = model.state_dict()   #模型状态字典
                if epoch%10 == 9:
                    save_network(model, epoch)   #每10轮保存一次模型
                draw_curve(epoch)  #画损失和误差曲线
        # 记录每一轮的训练时间
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))   #//表示整除
        print()
    #记录模型总训练时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(last_model_wts)
    save_network(model, 'last')   #保存训练完毕的名为net_last的模型
    return model

#---------------------------------4、画出训练和验证loss及error-------------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")    #分别画loss和top1的error
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)    #记录轮数，即横坐标
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')  #'bo-'是折线图的样式
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()  #标签框
        ax1.legend()
    fig.savefig( os.path.join('/content/GPUID/model',name,'train.jpg'))   #保存图像

#---------------------------------5、保存训练的模型------------------------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('/content/GPUID/model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)  #保存为CPU版本的模型参数
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])      #模型回到gpu

#---------------------------------6、设置网络、加载模型--------------------------------------
#这里加入了len(classname)，可以根据数据集判断多少类，从而适用不同的数据集
model = ft_net(len(class_names), opt.droprate, opt.stride)  #载入网络
if opt.PCB:
    model = PCB(len(class_names))
opt.nclasses = len(class_names)  #记录分类数
print(model)   #输出模型结构
#如果不用PCB
if not opt.PCB:
    ignored_params = list(map(id, model.classifier.parameters() ))  #id用于获取对象内存地址
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)    #调用SGD优化方法
    #weight_decay：限制自由参数数量防止过拟合
else:
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() ))
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},      #各层可以采用不同的学习率，parts=6，一共六层
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
             {'params': model.classifier4.parameters(), 'lr': opt.lr},
             {'params': model.classifier5.parameters(), 'lr': opt.lr},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

#每40轮学习率减少0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

#---------------------------------7、主程序，训练、评估--------------------------------------
dir_name = os.path.join('/content/GPUID/model',name)     #模型保存的位置
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#保存一份训练程序和模型程序
copyfile('/content/GPUID/train.py', dir_name+'/train.py')
copyfile('/content/GPUID/model.py', dir_name+'/model.py')

# 保存参数文件
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)  #将参数写入yaml文件，去掉大括号

#用gpu训练
model = model.cuda()
criterion = nn.CrossEntropyLoss()      #交叉熵损失函数
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=60)  #训练60轮

