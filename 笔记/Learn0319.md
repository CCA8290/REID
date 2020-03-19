# 3月19号学习
## 1 任务
1. 看一遍其他PCB源码
2. 过一遍自己的代码
3. 复习PCB结构，BN，dropout等知识
## 2 Python学习
### 2.1 nn.Conv2d卷积
>参考：[CSDN](https://blog.csdn.net/qq_26369907/article/details/88366147)，[博客园](https://www.cnblogs.com/siyuan1998/p/10809646.html)    
>&ensp;**例子**
>```python
>import torch
>import torch.nn as nn
>x = torch.randn(10, 16, 30, 32) # batch, channel , height , width
>print(x.shape)
>m = nn.Conv2d(16, 33, (3, 2), (2,1))  # in_channel, out_channel ,kennel_size,stride
>print(m)
>y = m(x)
>print(y.shape)
>```
>&ensp;**输出**
>```python
>>>torch.Size([10, 16, 30, 32])
>>>Conv2d(16, 33, kernel_size=(3, 2), stride=(2, 1))
>>>torch.Size([10, 33, 14, 31])
>```
>&ensp;**函数说明**   
>```python
>Torch.nn.Conv2d(in_channels，out_channels，kernel_size，stride=1，padding=0，dilation=1，groups=1，bias=True)
>```
>in_channels：输入维度    
>out_channels：输出维度     
>kernel_size：卷积核大小     
>stride：步长大小    
>padding：补0    
>dilation：kernel间距   
>&ensp;**计算方法**     
>![Conv2d公式](https://raw.githubusercontent.com/CCA8290/Pic_save/master/200317/conv2d.png)    
### 2.2 map()
>``map()`` 会根据提供的函数对指定序列做映射。   
>第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。    
>``map(function, iterable, ...)``    
>Python 3.x 返回迭代器    
>&ensp;**例子** 
>```python
>#!!!注意，以下结果是在python2中显示的
>def square(x) :            # 计算平方数
>    return x ** 2
>map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
>>> [1, 4, 9, 16, 25]
>map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
>>> [1, 4, 9, 16, 25]
># 提供了两个列表，对相同位置的列表数据进行相加
>map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
>>> [3, 7, 11, 15, 19]   
>#---------------------------------------------------- 
>#!!!在python3中返回的是迭代器
>print(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))
>>> <map object at 0x0000022436A31988>
>print(list(map(lambda x: x ** 2, [1, 2, 3, 4, 5])))
>>> [1,4,9,16,25]  #新的列表，并不影响原来的列表
>```
### 2.3 filter()
>``filter()`` 函数用于过滤序列，过滤掉不符合条件的元素，**返回**由符合条件元素组成的**新列表**。   
>该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。  
>``filter(function, iterable)``     
>&ensp;**例子** 
>```python
>#和上述map类似，python3也返回迭代器，下为python2
>def is_odd(n):  #过滤所有的奇数
>   return n % 2 == 1
>newlist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>print(newlist)
>>> [1, 3, 5, 7, 9]
>```    
### 2.4 optim.SGD
>参考：[简书](https://www.jianshu.com/p/ff0059a9d2cb),[CSDN1](https://blog.csdn.net/qq_34690929/article/details/79932416),[CSDN2](https://blog.csdn.net/lanran2/article/details/50409507)    
>``model.parameters()``是获取model网络的参数  
>构建好神经网络后，网络的参数都保存在``parameters()``函数当中
>用法示例：
>```python
>optimizer = optim.Adam([var1, var2], lr = 0.0001)
>```
>&ensp;**Momentum**    
>>Momentum 传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 此方法比较曲折。  
>> 
>>我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走, 走的弯路也变少了. 这就是 Momentum 参数更新。    
>>**冲量**这个概念源自于物理中的力学，表示力对时间的积累效应。在普通的梯度下降法x+=v中，每次x的更新量v为v=−dx∗lr，其中dx为目标函数func(x)对x的一阶导数。当使用冲量时，则把每次x的更新量v考虑为本次的梯度下降量−dx∗lr与上次x的更新量v乘上一个介于[0,1]的因子momentum的和，即：***v=-dx\*lr+v\*momentum***   
>>当本次梯度下降- dx * lr的方向与上次更新量v的方向相同时，上次的更新量能够对本次的搜索起到一个正向加速的作用。
>>当本次梯度下降- dx * lr的方向与上次更新量v的方向相反时，上次的更新量能够对本次的搜索起到一个减速的作用。    
>>**优点**：对于梯度改变方向的维度减少更新，对于梯度相同方向的维度增加更新。    
>>**缺点**:先计算坡度，然后进行大跳跃，盲目的加速下坡   
>>**改进**：Nesterov加速梯度法(NAG)
>   
>&ensp;**Learning rate**    
>>学习率较小时，收敛到极值的速度较慢。
>>学习率较大时，容易在搜索过程中发生震荡。
>
>&ensp;**weight decay**
>>为了有效限制模型中的自由参数数量以避免过度拟合，可以调整成本函数。   
>>一个简单的方法是通过在权重上引入零均值高斯先验值，这相当于将代价函数改变为E(w)= E(w)+λ2w2。
>>在实践中，这会惩罚较大的权重，并有效地限制模型中的自由度。正则化参数λ决定了如何将原始成本E与大权重惩罚进行折衷。   
>
>&ensp;**learning rate decay**
>>在使用梯度下降法求解目标函数func(x) = x * x的极小值时，更新公式为x += v，其中每次x的更新量v为v = - dx * lr，dx为目标函数func(x)对x的一阶导数。可以想到，如果能够让lr随着迭代周期不断衰减变小，那么搜索时迈的步长就能不断减少以减缓震荡。学习率衰减因子由此诞生：  
>>***lri=lrstart∗1.0/(1.0+decay∗i)***   
>>decay越小，学习率衰减地越慢，当decay = 0时，学习率保持不变。  
>>decay越大，学习率衰减地越快，当decay = 1时，学习率衰减最快。