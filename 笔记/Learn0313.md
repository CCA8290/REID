# 3月13号学习
## 1 任务
1. 继续读懂``evaluate.py``和``demo.py``     
2. 抓紧开始PyQt编程

## 2 Python学习
1. ``np.argwhere()``返回的是二维数组
2. ``np.in1d(a,b)``是判断a中是否有值存在于b，返回的是len(a)的bool型列表
### 2.1 PyQt学习
参考：[简书](https://www.jianshu.com/p/5b063c5745d0)     
1. 窗口一般用MainWindow或者Widget
2. 用Qtdesigner设计完界面将``.ui``文件编译为``.py``
3. 上述``.py``文件运行不能显示窗口，因为没定义程序入口，可编写一个新脚本调用，也可以在``.py``创建新的类。见下面的例子:
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from 编译好的py import *


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())

```
