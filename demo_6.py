import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#我的电脑plt不能显示中文字符，加入如下
plt.rcParams['font.sans-serif']=['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   # 用来正常显示负号

#---------------------------------1、设置参数，载入query和gallery-----------------------------
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--test_dir',default='C:/Users/CCA82/Desktop/Reid_origin/Market/pytorch',type=str, help='./test_data')
opts = parser.parse_args()
data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#---------------------------------2、读取结果，分类图片---------------------------------------
result = scipy.io.loadmat('C:/Users/CCA82/Desktop/Reid_origin/pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

#======图片分类可直接借用evaluate中筛选index的部分======
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    #求距离
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #处理index
    index = np.argsort(score)  #从小到大记录索引
    index = index[::-1]        #逆序
    #筛选index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)  #相机id相同
    #记录垃圾index
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 
    #去掉index中的垃圾标签
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index
#根据图片取得图片在mat中的索引
pic='0145_c3s1_023426_00.jpg'
for i in range(len(query_label)):
    pic_path,_=image_datasets['query'].imgs[i]
    if(pic==os.path.basename(pic_path)):
        break
print('在结果mat中的index为%d'%i)
#获得排序索引
index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

#---------------------------------3、可视化结果，保存图片-------------------------------------
def imshow(path, title=None):
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  

query_path, _ = image_datasets['query'].imgs[i]  #返回路径和label
#和上述label不同，这里是图片的真label，上述是文件类别在query中的顺序
#如：0319_c5s1_072148_00.jpg，上面返回160，下面返回319
query_label = query_label[i]  
print('query图片的路径为：',query_path)
print('识别前十张图片为:\n')
#绘制排序结果
#这里加异常处理是为了应对某些没有GUI程序如Qt，tkinter的
#linux系统而无法输出图片的情况
try: 
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)    #1行11列
    ax.axis('off')   #不显示坐标轴
    
    imshow(query_path,'query图片')   
    for i in range(10):    #找出排序前十的
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')     
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        print('图片%d为：'%(i+1),os.path.basename(img_path))
        print('query标签：',query_label,'   gallery标签：',label)
        if label == query_label:  #如果标签正确，绿色
            print('匹配正确')
            ax.set_title('%d cam_id:%d'%(i+1,gallery_cam[index[i]]), color='green')
        else:                     #否则红色
            print('匹配错误！')
            ax.set_title('%d'%(i+1), color='red')
        print()
except RuntimeError:
    print('查看排序结果需要GUI支持！')
fig.savefig("show.png")
plt.show()
