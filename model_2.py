#训练之前可以运行一下该程序，查看模型结构
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models #从pytorch引入模型
from torch.autograd import Variable
#import pretrainedmodels
#导入一些预训练模型如Resnet50，见下面的site  
#https://pypi.org/project/pretrainedmodels/

#何恺明初始化方法，针对ReLU的初始化，xavier仅在tanh中表现得好
def weights_init_kaiming(m): 
    classname = m.__class__.__name__ #输出类名
    #print(classname)
    if classname.find('Conv') != -1:   #find为寻找是否存在某字符串，没有则返回-1
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') #旧版的pytorch可以用kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)      #将Tensor填充为常量，这里是将bias置0
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)  #正态分布初始化，均值1，方差0.02
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# 定义新的全连接层和分类层，以配合Resnet50或PCB适应数据集
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True):
        super(ClassBlock, self).__init__()
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]  #全连接层，输入输出为二维张量
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]  #批量归一化层
        if relu:
            add_block += [nn.LeakyReLU(0.1)]  #LeakyReLU层
        if droprate>0:       
            add_block += [nn.Dropout(p=droprate)]  #dropout层
        add_block = nn.Sequential(*add_block) # *取内容作为参数，如a=[x],*a=x
        add_block.apply(weights_init_kaiming)  #He初始化
        #最后的分类层
        classifier = []  
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):   #前向传播函数
        x = self.add_block(x)
        x = self.classifier(x)
        return x

#基于ResNet50
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))   #改为自适应平均池化
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

#定义baseline --PCB，也是算法的核心
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 #图片分割的块数为6
        model_ft = models.resnet50(pretrained=True) #PCB也是基于Resnet50的
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1)) #自适应平均池化层
        self.dropout = nn.Dropout(p=0.5)
        #去掉最后的下采样层，步长改为1即失效
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        #定义6个分类器
        for i in range(self.part):
            name = 'classifier'+str(i)
            #setattr用来设置属性值。其实就是动态定义名称。
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        #得到特征数量 batchsize*2048*self.part*1，如32*2048*6*1
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i]) #去掉维数为1的维度，变为32*2048，共part个32*2048
            name = 'classifier'+str(i)
            c = getattr(self,name) #获取属性值
            predict[i] = c(part[i])

        #y预测结果加和，模型里不需要用到，用于测试
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        #按照论文，去掉最后的采样层
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
    #测试的时候不用加上最后的分类层
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2)) #[32,2048,6]
        return y

if __name__ == '__main__':
#向定义的模型传入参数，并不用net.forward(xxx)来前向传播
#实际上，net(xxx)等价于net.forward(xxx)
#该段程序只是用来查看、测试model的，方便调试，并无实际用途
    net = ft_net(751, stride=1)
    #net.classifier = nn.Sequential() #分类器暂且置为一个空的容器
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('网络输出size:')
    print(output.shape)
