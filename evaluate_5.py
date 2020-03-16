import scipy.io
import torch
import numpy as np
import os

#---------------------------------1、寻找索引对应关系，计算mAP--------------------------------
#找索引
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)          #变为列向量
    #求距离
    score = torch.mm(gf,query)     #矩阵相乘，由于已经求出L2范数，相乘即为余弦距离
    score = score.squeeze(1).cpu() #去掉维数为1的维度
    score = score.numpy()          #转为numpy
    #处理index
    index = np.argsort(score)      #按照元素从小到大记录下标，size为测试集图片数量
    index = index[::-1]            #逆序，就表示元素从大到小的下标，size:19732
    #筛选index 
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    #计算差集，不去重
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True) #label相同但相机id不同
    junk_index1 = np.argwhere(gl==-1)  #无效索引，label为-1
    junk_index2 = np.intersect1d(query_index, camera_index)  #交集，label相同相机id也相同
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

#计算指标
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_() #len(gallery)
    if good_index.size==0:   #若为空
        cmc[0] = -1
        return ap,cmc

    #去掉无用下标
    #查询index中是否有与junk_index不同的值，返回bool
    #in1d默认是找相同的值，但是这里加了反转invert
    #所以这两句程序意义重大：剔除垃圾index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]    #返回这些值

    #与上面类似，但这里是找出good_index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()  #变成一维数组
    #求AP
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
    return ap, cmc

#---------------------------------2、主程序实现---------------------------------------------
result = scipy.io.loadmat('C:/Users/CCA82/Desktop/Reid_origin/pytorch_result.mat')  #加载保存的结果
query_feature = torch.FloatTensor(result['query_f'])      #query特征，[3368, 512]
query_cam = result['query_cam'][0]                        #query摄像头号,[3368,]
query_label = result['query_label'][0]                    #query标签，[3368，]
gallery_feature = torch.FloatTensor(result['gallery_f'])  #gallery特征,[19732, 512]
gallery_cam = result['gallery_cam'][0]                    #gallery摄像头号,[19732,]
gallery_label = result['gallery_label'][0]                #gallery标签
#以上数据都是二维，加[0]表示取一维
#query_feature = query_feature.cuda()                      #放入显存
#gallery_feature = gallery_feature.cuda()

print('query图片数量为：',len(query_label))
print('gallery图片数量为：',len(gallery_label))                              
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC/len(query_label) #求出平均的CMC(rank)
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

