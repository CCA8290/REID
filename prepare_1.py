#此程序要首个运行，作用是将数据集按照行人编号分类
import os
from shutil import copyfile
#数据集的路径
download_path = 'Market'

if not os.path.isdir(download_path):    #用来判断download_path是否为目录
    print('please change the download_path')
save_path = download_path + '/pytorch'  #分类后保存路径
if not os.path.isdir(save_path):        #判断保存路径是否存在
    os.mkdir(save_path)                 #不存在则创建
	
#-------------------------------1、query------------------------------------
#分离query集
query_path = download_path + '/query'   #原始数据集的路径
query_save_path = download_path + '/pytorch/query'    #保存路径
if not os.path.isdir(query_save_path):   #同样的，如果不存在则创建
    os.mkdir(query_save_path)

#os.walk用于递归遍历整个目录，包括文件下，文件夹下的目录，文件夹下所有文件，
#topdown表示优先遍历根目录下的文件
for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':     #读取jpg格式的图片文件
            continue
        ID  = name.split('_')        #按照下划线切开文件名
        src_path = query_path + '/' + name 
        dst_path = query_save_path + '/' + ID[0]  
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
		#按照行人ID分类放至指定文件夹 
        copyfile(src_path, dst_path + '/' + name) 

#---------------------------2、multi-query----------------------------------
#multi-query     多个query查询
query_path = download_path + '/gt_bbox'
#如果文件夹不存在则跳过，兼容不含multi-query的数据集，如Duke数据集
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#------------------------------3、gallery-----------------------------------
#将测试集图片按照ID放至gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#-----------------------------4、train_all----------------------------------
#分类所有的训练数据
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#------------------------------5、train&val---------------------------------
#从训练集里每人选取一张图作为验证集，剩余的作为实验训练集
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
#对于每个ID，第一次copy单个文件至val，之后由于val中某ID已经
#存在，故跳过上面的判断语句，dst变为train，直到遇到下一个ID
#需要注意的是，在整个程序中，src_path始终表示的是单个图片
