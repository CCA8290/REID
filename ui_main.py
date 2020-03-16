# -*- coding: utf-8 -*-
import sys
import os 
import time
import scipy.io
import torch
import numpy as np
from torchvision import datasets
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *   
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(680, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(680, 500))
        MainWindow.setMaximumSize(QtCore.QSize(1920, 1080))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pic_11 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pic_11.setFont(font)
        self.pic_11.setStyleSheet("background-color: transparent")
        self.pic_11.setAlignment(QtCore.Qt.AlignCenter)
        self.pic_11.setObjectName("pic_11")
        self.verticalLayout_4.addWidget(self.pic_11)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet("background-color: transparent")
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_4.addWidget(self.textEdit)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(64, 128))
        self.label.setMaximumSize(QtCore.QSize(64, 128))
        self.label.setStyleSheet("background:white;")
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(114, 28))
        self.pushButton.setMaximumSize(QtCore.QSize(200, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(114, 28))
        self.pushButton_2.setMaximumSize(QtCore.QSize(200, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setMinimumSize(QtCore.QSize(114, 28))
        self.pushButton_3.setMaximumSize(QtCore.QSize(200, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setMinimumSize(QtCore.QSize(255, 128))
        self.textBrowser.setMaximumSize(QtCore.QSize(310, 128))
        self.textBrowser.setStyleSheet("border-width: 1px;border-style: solid;border-color: rgb(135, 206, 250)")
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout.addWidget(self.textBrowser, 1, 0, 1, 2)
        self.horizontalLayout_3.addLayout(self.gridLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pic_0 = QtWidgets.QLabel(self.centralwidget)
        self.pic_0.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_0.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_0.setStyleSheet("background:white")
        self.pic_0.setText("")
        self.pic_0.setObjectName("pic_0")
        self.horizontalLayout.addWidget(self.pic_0)
        self.pic_1 = QtWidgets.QLabel(self.centralwidget)
        self.pic_1.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_1.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_1.setStyleSheet("background:white;")
        self.pic_1.setText("")
        self.pic_1.setObjectName("pic_1")
        self.horizontalLayout.addWidget(self.pic_1)
        self.pic_2 = QtWidgets.QLabel(self.centralwidget)
        self.pic_2.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_2.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_2.setStyleSheet("background:white;")
        self.pic_2.setText("")
        self.pic_2.setObjectName("pic_2")
        self.horizontalLayout.addWidget(self.pic_2)
        self.pic_3 = QtWidgets.QLabel(self.centralwidget)
        self.pic_3.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_3.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_3.setStyleSheet("background:white;")
        self.pic_3.setText("")
        self.pic_3.setObjectName("pic_3")
        self.horizontalLayout.addWidget(self.pic_3)
        self.pic_4 = QtWidgets.QLabel(self.centralwidget)
        self.pic_4.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_4.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_4.setStyleSheet("background:white;")
        self.pic_4.setText("")
        self.pic_4.setObjectName("pic_4")
        self.horizontalLayout.addWidget(self.pic_4)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pic_5 = QtWidgets.QLabel(self.centralwidget)
        self.pic_5.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_5.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_5.setStyleSheet("background:white;")
        self.pic_5.setText("")
        self.pic_5.setObjectName("pic_5")
        self.horizontalLayout_2.addWidget(self.pic_5)
        self.pic_6 = QtWidgets.QLabel(self.centralwidget)
        self.pic_6.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_6.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_6.setStyleSheet("background:white;")
        self.pic_6.setText("")
        self.pic_6.setObjectName("pic_6")
        self.horizontalLayout_2.addWidget(self.pic_6)
        self.pic_7 = QtWidgets.QLabel(self.centralwidget)
        self.pic_7.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_7.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_7.setStyleSheet("background:white;")
        self.pic_7.setText("")
        self.pic_7.setObjectName("pic_7")
        self.horizontalLayout_2.addWidget(self.pic_7)
        self.pic_8 = QtWidgets.QLabel(self.centralwidget)
        self.pic_8.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_8.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_8.setStyleSheet("background:white;")
        self.pic_8.setText("")
        self.pic_8.setObjectName("pic_8")
        self.horizontalLayout_2.addWidget(self.pic_8)
        self.pic_9 = QtWidgets.QLabel(self.centralwidget)
        self.pic_9.setMinimumSize(QtCore.QSize(64, 128))
        self.pic_9.setMaximumSize(QtCore.QSize(64, 128))
        self.pic_9.setStyleSheet("background:white;")
        self.pic_9.setText("")
        self.pic_9.setObjectName("pic_9")
        self.horizontalLayout_2.addWidget(self.pic_9)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 680, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pic=[self.pic_0,self.pic_1,self.pic_2,self.pic_3,self.pic_4,
            self.pic_5,self.pic_6,self.pic_7,self.pic_8,self.pic_9]    #记录所有的图片输出框(label)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "行人重识别"))
        self.pic_11.setText(_translate("MainWindow", "智能行人重识别系统"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'仿宋\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">使用说明：</p>\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1、可选择任意query图片，打开或拖拽均可</p>\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2、识别正确的图片边框为绿色，否则为红色</p>\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">3、识别结果从上到下从左到右依次排序</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Query图片"))
        self.pushButton.setText(_translate("MainWindow", "选择query"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "清空输出"))
        btn=self.pushButton
        btn.clicked.connect(self.openimage)  #打开图片按钮的功能
        btn_3=self.pushButton_3
        btn_3.clicked.connect(self.clear_output)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.pic_path=imgName      #保存图片路径
        #self.textBrowser.append('图片路径为:'+imgName)
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
    def clear_output(self):
        self.label.setPixmap(QtGui.QPixmap())
        for i in range(10):
            self.pic[i].setPixmap(QtGui.QPixmap())
            self.pic[i].setStyleSheet("background:white;")
        self.textBrowser.clear()

class Demo(QWidget,Ui_MainWindow):
    def __init__(self, parent=None):
        super(Demo, self).__init__(parent)
        self.data_dir = 'C:/Users/CCA82/Desktop/Reid_origin/Market/pytorch'
        self.image_datasets = {x: datasets.ImageFolder( os.path.join(self.data_dir,x) ) 
        for x in ['gallery','query']}

        self.result = scipy.io.loadmat('C:/Users/CCA82/Desktop/Reid_origin/pytorch_result.mat')
        self.query_feature = torch.FloatTensor(self.result['query_f'])
        self.query_cam = self.result['query_cam'][0]
        self.query_label = self.result['query_label'][0]
        self.gallery_feature = torch.FloatTensor(self.result['gallery_f'])
        self.gallery_cam = self.result['gallery_cam'][0]
        self.gallery_label = self.result['gallery_label'][0] 
        self.pic_path=''
    def sort_img(self,qf, ql, qc, gf, gl, gc):
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
    def get_index(self,pic_name):
        pic=os.path.basename(pic_name)
        for i in range(len(self.query_label)):
            pic_path,_=self.image_datasets['query'].imgs[i]
            if(pic==os.path.basename(pic_path)):
                break
        self.textBrowser.append('query在结果mat中的index为%d'%i)
        return i
 
    def reid(self):
        #获得排序索引
        i=self.get_index(self.pic_path)
        index = self.sort_img(self.query_feature[i],self.query_label[i],self.query_cam[i],
                        self.gallery_feature,self.gallery_label,self.gallery_cam)
        query_path, _ = self.image_datasets['query'].imgs[i]  #返回路径和label
        query_label = self.query_label[i] 
        self.textBrowser.append('query图片的路径为：'+query_path+'\n')
        self.textBrowser.append('识别前十张图片为:') 
        #绘制排序结果
        d={x: 'label_'+str(x) for x in range(10)}
        for i in range(10):    #找出排序前十的    
            img_path, _ = self.image_datasets['gallery'].imgs[index[i]]
            label = self.gallery_label[index[i]]
            jpg = QtGui.QPixmap(img_path).scaled(self.label.width(), self.label.height())
            self.pic[i].setPixmap(jpg)
            self.textBrowser.append('图片%d为：'%(i+1)+os.path.basename(img_path)) 
            self.textBrowser.append('query标签：'+str(query_label)+'   gallery标签：'+str(label)) 
            if label == query_label:  #如果标签正确，绿色
                self.textBrowser.append('匹配正确') 
                self.textBrowser.append('cam_id:%d\n'%(self.gallery_cam[index[i]])) 
                self.pic[i].setStyleSheet("border-width: 2px;border-style: solid;border-color: rgb(0, 255, 0)")
            else:                     
                self.textBrowser.append('匹配错误\n')
                self.pic[i].setStyleSheet("border-width: 2px;border-style: solid;border-color: rgb(255, 0, 0)")            
            #time.sleep(0.1)
#---------------------------------2、主窗口，继承上述控件类------------------------- 
class MyWindow(QMainWindow,Demo):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.setAcceptDrops(True)    #窗体接受拖拽操作
        btn_2=self.pushButton_2
        btn_2.clicked.connect(self.reid) 
    #=======设置拖拽操作========
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            try:
                event.setDropAction(Qt.CopyAction)
            except Exception as e:
                print(e)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        try:
            if event.mimeData().hasUrls:
                event.setDropAction(Qt.CopyAction)
                event.accept()
                links = []
                #获取拖拽文件的路径名，存入列表
                for url in event.mimeData().urls():
                    links.append(str(url.toLocalFile()))
                #从路径读入图片，在query框显示
                self.pic_path=links[0]      #保存图片路径
                jpg = QtGui.QPixmap(links[0]).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(jpg)
            else:
                event.ignore()
        except Exception as e:
            print(e) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())


