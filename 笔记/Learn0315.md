# 3月15-18号学习
## 1 任务
1. 学习Qt
2. 完成程序工作
3. 学习模型及相关知识(3月17)，如：BN、Dropout、PCB等

## 2 Python及Qt
### 2.1 Qt
>1.PyQt5学习-[布局管理](https://www.jianshu.com/p/3832eb48f3d5)   
>2.PyQt5学习-[实例](https://www.jianshu.com/p/61cb5ed4548f)   
### 2.2 模型、python学习
#### 2.2.1 初始化方法（xavier和kaiming）
>[Xavier](https://www.cnblogs.com/hejunlin1992/p/8723816.html)适合tanh函数（线性激活函数），但用Relu表现差，[何恺明提出了针对Relu的初始化方法](https://www.jb51.net/article/167914.htm)。Pytorch默认使用**kaiming正态分布**初始化卷积层参数。   
[对比异同](https://blog.csdn.net/xxy0118/article/details/84333635)   
``torch.init``中的初始化方法，这篇[博客](https://blog.csdn.net/dss_dssssd/article/details/83959474)对比了几种方法

>**Xavier初始化**  
>使得每一层输出的方差应该尽量相等
>1. xavier均匀分布
>```python
>torch.nn.init.xavier_uniform_(tensor, gain=1)
>```   
>![xavier1](https://raw.githubusercontent.com/CCA8290/Pic_save/master/200317/xavier_uniform.jpg)  
>&emsp;例子:
>```python
>w = torch.empty(3, 5)
>nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
>```
>2. xavier正态分布
>```python
>torch.nn.init.xavier_normal_(tensor, gain=1)
>```   
>![xavier2](https://raw.githubusercontent.com/CCA8290/Pic_save/master/200317/xavier_normal.jpg) 
>**kaiming初始化**           
>1. kaiming均匀分布
>```python
>torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
>```
>![he1](https://raw.githubusercontent.com/CCA8290/Pic_save/master/200317/he_uniform.jpg) 
>也被称为 He initialization。   
>**a** – the negative slope of the rectifier used after this layer (0 for ReLU by default).激活函数的负斜率。   
>**mode** – either ‘fan_in' (default) or ‘fan_out'. Choosing fan_in preserves the magnitude of the variance of the weights in the forward pass. Choosing fan_out preserves the magnitudes in the backwards pass.        
>默认为fan_in模式，``fan_in``可以保持**前向传播**的权重方差的数>量级，``fan_out``可以保持**反向传播**的权重方差的数量级。           
>&emsp;2. kaiming正态分布
>```python
>torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
>```
>![he2](https://raw.githubusercontent.com/CCA8290/Pic_save/master/200317/he_normal.jpg)   
#### 2.2.2 PyTorch前向传播函数forward
>参考：[CSDN](https://blog.csdn.net/u011501388/article/details/84062483)
>1. 定义可学习参数的网络结构（堆叠各层和层的设计）；
>2. 数据集输入；
>3. 对输入进行处理（由定义的网络层进行处理）,主要体现在网络的前向传播；
>4. 计算loss ，由Loss层计算；
>5. 反向传播求梯度；
>6. 根据梯度改变参数值,最简单的实现方式（SGD）为:   
   ``weight = weight - learning_rate * gradient``
>7. 向定义的模型传入参数，并不用net.forward(xxx)来前向传播,实际上，net(xxx)等价于net.forward(xxx)
#### 2.2.3 PyTorch nn.Linear()
>PyTorch的``nn.Linear()``是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，一般形状为：    
>[batch_size, size]，不同于卷积层要求输入输出是四维张量。
>
>**in_features**指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。   
>
>**out_features**指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
>
>从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。  
>    
>&ensp;**例子**：
>```python
>import torch as t
>from torch import nn
># in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
>connected_layer = nn.Linear(in_features = 64*64*3, out_features = 2)
># 假定输入的图像形状为[64,64,3]
>input = t.randn(1,64,64,3)
># 将四维张量转换为二维张量之后，才能作为全连接层的输入
>input = input.view(1,64*64*3)
>print(input.shape)
>output = connected_layer(input) # 调用全连接层
>print(output.shape)
>```
>&ensp;运行结果：
>```
>torch.Size([1, 12288])
>torch.Size([1, 2])
>```   
#### 2.2.4 PyTorch view()
>&ensp;**例子**
>```python
>a=torch.Tensor([[[1,2,3],[4,5,6]]])
>b=torch.Tensor([1,2,3,4,5,6])
>print(a.view(1,6))
>print(b.view(1,6))
>```
>&ensp;输出：(类似reshape)
>```
>tensor([[1., 2., 3., 4., 5., 6.]]) 
>tensor([[1., 2., 3., 4., 5., 6.]]) 
>```
