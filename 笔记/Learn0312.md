# 3月12号学习
<!-- TOC -->

- [3月12号学习](#3月12号学习)
    - [任务内容](#任务内容)
    - [python学习](#python学习)
        - [torch.squeeze()和torch.unsqueeze()用法](#torchsqueeze和torchunsqueeze用法)
        - [np.argsort()](#npargsort)
        - [index[::-1]是逆序切片](#index-1是逆序切片)
        - [np.setdiff1d](#npsetdiff1d)
    - [ReID评估标准](#reid评估标准)
        - [1. rank-n](#1-rank-n)
        - [2.Precision & Recall](#2precision--recall)
        - [3、mAP](#3map)
        - [4、CMC](#4cmc)
        - [5、ROC](#5roc)
        - [6、PR曲线](#6pr曲线)

<!-- /TOC -->
## 任务内容    
1. 读懂 ``evalute.py``和``demo.py``
2. 尝试界面编程
---
## python学习
### torch.squeeze()和torch.unsqueeze()用法 
    
    torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，默认是将a中所有为1的维度删掉。也可以通过dim指定位置，删掉指定位置的维数为1的维度。

    torch.unsqueeze()这个函数主要是对数据维度进行扩充。需要通过dim指定位置，给指定位置加上维数为1的维度。

###  np.argsort()     
**将numpy数组中元素从小到大排列并输出索引**
```python
a=np.array([3,2,-1,7,6])
print(np.argsort(a))
>>[2 1 0 4 3]
```
### index[::-1]是逆序切片

### np.setdiff1d 
``setdiff1d(ar1, ar2, assume_unique=False)``           

    1.功能：找到2个数组中集合元素的差异。
    2.返回值：在ar1中但不在ar2中的已排序的唯一值。

    3.参数：
    ar1：array_like 输入数组。
    ar2：array_like 输入比较数组。
    assume_unique：bool。如果为True，则假定输入数组是唯一的，即可以加快计算速度。 默认值为False
+ assume_unique = False的情况：
```python
    a = np.array([8,2,3,2,4,1])
    b = np.array([7,4,5,6,3])
    c = np.setdiff1d(a, b)
    print(c)#[1 2 8]
```
+ assume_unique = True的情况：
```python
    a = np.array([8,2,3,2,4,1])
    b = np.array([7,4,5,6,3])
    c = np.setdiff1d(a, b,True)
    print(c)#[8 2 2 1]
```
``np.intersect1d``计算交集，和setdiff1d参数类似，后者实际是求差集

---

## ReID评估标准
来源：[CSDN](https://blog.csdn.net/weixin_40446557/article/details/83106995)
在此之前，看一下对行人重识别的[解释](https://blog.csdn.net/ctwy291314/article/details/83618646)
>在毕设代码**evaluate_gpu.py**中，计算的思路如下：
>遍历每个query，对于每个query，找到gallery中跟当前query属于同一个ID但是是不同camera下的图片，这些图片被认为是good_image，其他的图片就是junk_image，然后对所有gallery与当前query的距离做个升序排序，接着就对当前query计算AP，cmc(rank)
### 1. rank-n
**搜索结果中最靠前（置信度最高）的n张图有正确结果的概率。**  

    例如： lable为 m1，在100个样本中搜索。    

    如果识别结果是 m1、m2、m3、m4、m5……，则此时rank-1的正确率
    为100%；rank-2的正确率也为100%；rank-5的正确率也为100%；

    如果识别结果是 m2、m1、m3、m4、m5……，则此时rank-1的正确率
    为0%；rank-2的正确率为100%；rank-5的正确率也为100%；

    如果识别结果是 m2、m3、m4、m5、m1……，则此时rank-1的正确率
    为0%；rank-2的正确率为0%；rank-5的正确率为100%

### 2.Precision & Recall
参考：[CSDN-1](https://blog.csdn.net/u012879957/article/details/80564148),[简书](https://www.jianshu.com/p/4434ea11c16c)
>**TP : True Positive 预测为1，实际也为1  
>TN：True Nagetive 预测为0，实际也为0   
>FP：False Positive 预测为1，实际为0的   
>FN：False Nagetive 预测为0，实际为1的**          
> |预测\实际|正|负|
> |:-:|:-:|:-:|
> |正|TP|FP|
> |负|FN|TN|
[^_^]:(注释，由于github不支持公式，我把Latex存在这里$$\frac{TP+TN}{TP+TN+FP+FN}$$,$$\frac{TP}{TP+FP}$$,$$\frac{TP}{TP+FN}$$,$$F_1=\frac{2*P*R}{P+R}$$,$$F_1=\frac{2TP}{2TP+FP+FN}$$)
<!--也可以用HTML的注释哦-->
><span style=color:red>Accuracy：准确率</span>
>预测正确的数量          
>![公式](https://raw.githubusercontent.com/CCA8290/Pic_save/master/Latex/1.jpg)     
><font color=red>Precision：精确率</font>
>它表示的是预测为正的样本(TP,FP)中有多少是真正的正样本        
>![公式](https://raw.githubusercontent.com/CCA8290/Pic_save/master/Latex/2.jpg)      
><font color=red>Recall：召回率</font>
>表示样本中的正例(TP,FN)有多少被预测正确了      
>![公式](https://raw.githubusercontent.com/CCA8290/Pic_save/master/Latex/3.jpg)       
><font color=red>F-score，也叫F<sub>1</sub>值</font>
>是精确率和召回率的调和均值     
>![公式](https://raw.githubusercontent.com/CCA8290/Pic_save/master/Latex/4.jpg)      
>同样地：    
>![公式](https://raw.githubusercontent.com/CCA8290/Pic_save/master/Latex/5.jpg)   
### 3、mAP
>AP(平均Precision，即平均精度)衡量的是学出来的模型在单个类别上的好坏。mAP衡量的是学出的模型在所有类别上的好坏，也即**PR曲线**线下的面积（PR曲线: 所有样本的precision和recall绘制在图里）表示    
![mAP1](https://raw.githubusercontent.com/CCA8290/Pic_save/master/REID_mAP.jpg)
![mAP2](https://raw.githubusercontent.com/CCA8290/Pic_save/master/REID_mAP2.png)
### 4、CMC
>CMC曲线是算一种**top-k**的击中概率，主要用来评估闭集中rank的正确率
>例如待识别人脸有3个（假如label为m1，m2，m3），同样对每一个人脸都有一个从高到低的得分。  

>比如人脸1结果为m1、m2、m3、m4、m5……，人脸2结果为m2、m1、m3、m4、m5……，人脸3结果m3、m1、m2、m4、m5……，则此时rank-1的正确率为（1+1+1）/3=100%；rank-2的正确率也为（1+1+1）/3=100%；rank-5的正确率也为（1+1+1）/3=100%；    

>比如人脸1结果为m4、m2、m3、m5、m6……，人脸2结果为m1、m2、m3、m4、m5……，人脸3结果m3、m1、m2、m4、m5……，则此时rank-1的正确率为（0+0+1）/3=33.33%；rank-2的正确率为（0+1+1）/3=66.66%；rank-5的正确率也为（0+1+1）/3=66.66%；
><span style=color:blue>PS：和rank好像，我在此时间点还未搞清楚</span>
### 5、ROC

>**ROC曲线是检测、分类、识别任务中很常用的一项评价指标**    
>ROC曲线上的每一点反映的是不同的阈值对应的FP（false positive）和TP（true positive）之间的关系     
>![ROC曲线](https://raw.githubusercontent.com/CCA8290/Pic_save/master/ROC.jpg)      
>通常情况下，ROC曲线越靠近（0，1）坐标表示性能越好。   
>**TPR=TP/(TP+FN)=Recall**    
>**FPR=FP/(FP+TN)，预测为好人，实际是坏人的占所有坏人的比例**   

### 6、PR曲线
>参考：  
[PR曲线深入理解](https://blog.csdn.net/b876144622/article/details/80009867)    
[机器学习PRC](https://blog.csdn.net/weixin_31866177/article/details/88776718)    
>**PR曲线就是<span style=color:red>精确率precision vs 召回率recall</span> 曲线，以recall作为横坐标轴，precision作为纵坐标轴。**
>![PR曲线](https://raw.githubusercontent.com/CCA8290/Pic_save/master/PRC.jpg)