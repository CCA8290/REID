# 3月14号学习
## 1 任务
1. 看完``demo.py``
2. 设计可视化程序
## 2 学习内容
### 2.1 [Market-1501](https://blog.csdn.net/ctwy291314/article/details/83544088)数据集复习
>6个Camera，5个HD一个普通      
>1501个行人，32668个行人矩形图        
>Training_set 751人-12936张图像       
>Test_set 750人-19732张图像         
>Query 3368张，由人工标记          
>Gallery 用DPM检测器检测  

>**文件目录**  
├── Market/
│   ├── bounding_box_test//* 用于测试的图像      
│   ├── bounding_box_train//* 用于训练的图像    
│   ├── gt_bbox//*本次实验用不到    
│   ├── gt_query//* 用于multiple query testing       
│   ├── query//* query的图片        
│   ├── readme.txt       
>**bounding_box_test**        
750人的测试集      
前缀0000表示DPM检测错的图，-1表示不属于这750个人的图    
>以 ``0001_c1s1_000151_01.jpg`` 为例   
>+ 0001 表示每个人的标签编号，从0001到1501；   
>+ c1 表示第一个摄像头(camera1)，共有6个摄像头；   
>+ s1 表示第一个录像片段(sequece1)，每个摄像机都有数个录像段；   
>+ 000151 表示 c1s1 的第000151帧图片，视频帧率25fps；   
>+ 01 表示 c1s1_001051 这一帧上的第1个检测框，由于采用DPM检测器，对于每一帧上的行人可能会框出好几个bbox。00 表示手工标注框。  
### 2.2 matplotlib无法正常显示中文
>加入如下语句：
>```python
>plt.rcParams['font.sans-serif']=['SimHei']   # 用来正常显示中文标签
>plt.rcParams['axes.unicode_minus']=False   # 用来正常显示负号
>```           
### 2.3 Python学习
>1. ``os.path.basename(file_path)``可以读取路径最后的文件名，如：``C:\Desktop\a.txt``，返回值为**a.txt**