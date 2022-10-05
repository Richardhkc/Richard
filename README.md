# U—NET（盐矿挑战） 学习记录
## 自我情况介绍：
目前就读于日本上智大学，修士一年，主要研究方向是图像处理方向，写这个md文件的想法是记录一下自己的学习历程，顺便可以复盘一下。
## 导入库
``` Python
    import os
    import numpy as np
    import pandas as pd
    import cv2
    import matplotlib.pyplot as plt
```
### OS
os就是“operating system”的缩写，顾名思义，os模块提供的就是各种 Python 程序与操作系统进行交互的接口。通过使用os模块，一方面可以方便地与操作系统进行交互，另一方面页可以极大增强代码的可移植性。如果该模块中相关功能出错，会抛出OSError异常或其子类异常。
### numpy
NumPy是Python的一个用于科学计算的基础包。它提供了多维数组对象，多种衍生的对象（例如隐藏数组和矩阵）和一个用于数组快速运算的混合的程序，包括数学，逻辑，排序，选择，I/O，离散傅立叶变换，基础线性代数，基础统计操作，随机模拟等等。
### pandas
pandas 是基于NumPy 的一种工具，该工具是为解决数据分析任务而创建的。Pandas 纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。pandas提供了大量能使我们快速便捷地处理数据的函数和方法。
### cv2
OpenCV的全称是：Open Source Computer Vision Library。OpenCV是一个基于（开源）发行的跨平台计算机视觉库，可以运行在Linux、Windows和Mac OS操作系统上。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。
### matplotlib
Matplotlib 可能是 Python 2D-绘图领域使用最广泛的套件。它能让使用者很轻松地将数据图形化，并且提供多样化的输出格式。
## config(配置)
``` Python
n_fold = 5 # k折交叉验证，这里取5的意思是利用五折交叉验证。
#步骤是：1、将所有数据集分成五份，不重复地每次取其中一份做测试集，用其他四份做训练集训练模型，之后计算在该模型在测试集上的MSEi（均方误差，方差，Mean squared error）MSE越小，说明预测模型描述实验数据具有更好的精确度。
pad_left = 27 # 左部补零列数（padding规则，vaild：边缘不补充。Same：边缘补充）
pad_right = 27 # 右部补零列数
fine_size = 202 # 与pad-left和right加一起等于256，应该是图片像素？
batch_size = 18 # 批大小/批尺寸 batch_size：即一次训练所抓取的数据样本数量。batch_size将影响到模型的优化程度和速度。
epoch = 300 # 时期，一个时期=所有训练样本的一个正向传递和一个反向传递。（单次epoch=（全部训练样本/batchsize） / iteration（迭代） =1）
snapshot = 6 # 每六轮保存一次模型，防止停电中断训练。
max_lr = 0.012 # CLR（学习率周期）不是单调地降低训练过程中的学习率，而是让学习率在设定好地最大值与最小值之间往复变化
min_lr = 0.001 # 学习率最小值
momentum = 0.9 # deep learning最优化方法——动量
# 1.动量方法主要是为了解决Hessian矩阵病态条件问题（直观上讲就是梯度高度敏感于参数空间的某些方向）的。2.加速学习 3.一般将参数设为0.5,0.9，或者0.99，分别表示最大速度2倍，10倍，100倍于SGD的算法。
weight_decay = 1e-4 #权重衰减，L2正则化的目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题，所以权重衰减也叫L2正则化。

n_fold = 5
device = torch.device('cuda') # 表示将构建的张量或者模型分配到相应的设备上。
save_weight = 'weights/' # 保存模型
if not os.path.isdir(save_weight):
  os.mkdir(save_weight)
weight_name = 'model_' + str(fine_size+pad_left+pad_right) + '_res18' 
# os.path.isdir(xxx)判断某个路径是否为目录。
# os.mkdir（xxx）创建目录 
# str（）函数讲其他数据类型转化为字符串。
train_image_dir = 'tgs-salt-identification-challenge/train/images'
train_mask_dir = 'tgs-salt-identification-challenge/train/masks'
test_image_dir = 'tgs-salt-identification-challenge/test/images'
``` 
## padding规则
### 输入的图像的尺寸高和宽定义成：in_height，in_width
#### 卷积核的高和宽定义成filter_height、filter_width
#### 输入的尺寸中高和宽定义成output_height、out_width
#### 步长的高宽方向定义成strides_height、strides_width.
---
## VALID情况：边缘不填充

  输入宽和高的公式分别为：
* output_width=(in_width-filter_width+1)/strides_width(结果向上取整)
* output_height=(in_height-filter_height+1)/strides_height(结果向上取整)
--- 
## SAME情况：边缘填充
#### 输出的宽和高将与卷积核没关系，具体公式如下：
* output_width=in_width/strides_width(结果向上取整)
* output_height=in_height/strides_height(结果向上取整)

* 这里有一个很重要的知识点——补零的规则，见如下公式：

pad_height=max((out_height-1)*strides_height

+filter_height-in_height,0)

pad_width=max((out_width-1)*strides_width+filter_width-in_width,0)

pad_top=pad_height/2

pad_bottom=pad_height-pad_top

pad_left=pad_width/2

pad_right=pad_width-pad_left

上面公式中

pad_height:代表高度方向上要填充0的行数。

pad_width:代表宽度方向要填充0的列数。

pad_top、pad_bottom、pad_left、pad_right分别代表上、下、左、右这4个方向填充0的行、列数。

## Split（分割）
### 代码
```Python
depths = pd.read_csv('tgs-salt-identification-challenge/depths.csv') #读取数据
depths.sort_values('z', inplace=True) # 把z排序
depths.drop('z', axis=1, inplace=True) # 去掉z
depths['fold'] = (list(range(0,5)) * depths.shape[0])[:depths.shape[0]] # 0-4循环22000次后取长度到22000。

train_df = pd.read_csv('tgs-salt-identification-challenge/train.csv')
train_df = train_df.merge(depths)
dist = []
for id in train_df.id.values:
  img = cv2.imread(f'tgs-salt-identification-challenge/train/images/{id}.png', cv2.IMREAD_GRAYSCALE)
  dist.append(np.unique(img).shape[0])
train_df['unique_pixels'] = dist
```
![0-4循环22000次后取长度到22000](https://github.com/Richardhkc/Richard/blob/main/Learn%20history/22000times.png?raw=true)


