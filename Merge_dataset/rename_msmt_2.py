# -*- coding:utf-8 -*-
#MSMT17中训练集和测试集索引均从0开始，要对其重命名
import os
import shutil

def rename(src_path,dst_path):
    img_names = os.listdir(src_path)
    for img_name in img_names:
        if '.jpg' not in img_name:
            continue
        temp_name=int(img_name[0:4  ])
        temp_name+=1502
        print(img_name)
        #print(temp_name)
        #print(str(temp_name).zfill(4))
        srcFile=os.path.join(src_path,img_name)
        dstFile=os.path.join(dst_path,str(temp_name).zfill(4)+img_name[4:])
        #os.rename(srcFile,dstFile)   #OSError: [WinError 17] 系统无法将文件移到不同的磁盘驱动器
        shutil.copy(srcFile,dstFile)

if __name__ == '__main__':
    #源路径
    src_train_path = r'C:\Users\CCA82\Desktop\Reid_origin\MSMT17\bounding_box_train'
    src_query_path = r'C:\Users\CCA82\Desktop\Reid_origin\MSMT17\query'
    src_test_path = r'C:\Users\CCA82\Desktop\Reid_origin\MSMT17\bounding_box_test' 
    # 将整个Market数据集作为训练集
    #目标路径
    dst_dir = r'D:\REID\Market\bounding_box_train'
 
    rename(src_train_path, dst_dir)
    rename(src_query_path, dst_dir)
    rename(src_test_path, dst_dir)
