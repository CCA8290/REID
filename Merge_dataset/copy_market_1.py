import os
import re
import shutil

def make_market_dir(dst_dir='./'):
    market_root = os.path.join(dst_dir, 'Market')
    train_path = os.path.join(market_root, 'bounding_box_train')
    query_path = os.path.join(market_root, 'query')
    test_path = os.path.join(market_root, 'bounding_box_test')
 
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(query_path):
        os.makedirs(query_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
 
 def extract_market(src_path, dst_dir):
    img_names = os.listdir(src_path)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for img_name in img_names:
        if '.jpg' not in img_name:
            continue
        print(img_name)
        # pid: 每个人的标签编号 1
        # _  : 摄像头号 2
        pid, _ = map(int, pattern.search(img_name).groups())
        # 去掉没用的图片
        if pid == 0 or pid == -1:
            continue
        shutil.copy(os.path.join(src_path, img_name), os.path.join(dst_dir, img_name))

if __name__ == '__main__':
    make_market_dir(dst_dir='D:/reID')
    #源路径
    src_train_path = r'C:\Users\CCA82\Desktop\Reid_origin\Market\bounding_box_train'
    src_query_path = r'C:\Users\CCA82\Desktop\Reid_origin\Market\query'
    src_test_path = r'C:\Users\CCA82\Desktop\Reid_origin\Market\bounding_box_test' 
    # 将整个Market数据集作为训练集
    #目标路径
    dst_dir = r'D:\REID\Market\bounding_box_train'
 
    extract_market(src_train_path, dst_dir)
    extract_market(src_query_path, dst_dir)
    extract_market(src_test_path, dst_dir)