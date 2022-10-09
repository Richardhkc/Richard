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
train_df = train_df.merge(depths) # 合并，基于指定列的横向合并拼接。
dist = [] # 字典
for id in train_df.id.values:
  img = cv2.imread(f'tgs-salt-identification-challenge/train/images/{id}.png', cv2.IMREAD_GRAYSCALE)
  #cv2.IMREAD_GRAYSCALE以灰度模式读取图像
  dist.append(np.unique(img).shape[0]) # append 在列表末尾添加新的对象。np.unique函数去除其中重复的元素,并从小到大排列。（提取图像特征值）
train_df['unique_pixels'] = dist
```
<img src="https://github.com/Richardhkc/Richard/blob/main/Learn%20history/22000times.png?raw=true" width="100px">

## dataset
## 代码
``` Python
def trainImageFetch(images_id):
  image_train = np.zeros((images_id.shape[0], 101, 101), dtype=np.float32)#这里的np.float32是将改变数值数据类型。np.zeros(shape,dtype=float,order='C')
  mask_train = np.zeros((images_id.shape[0], 101, 101), dtype=np.float32)

  for idx, image_id in tqdm(enumerate(images_id), total=images_id.shape[0]):#shape【0】表示输出行数
    image_path = os.path.join(train_image_dir, image_id+'.png')#设置文件保存路径
    mask_path = os.path.join(train_mask_dir, image_id+'.png')#设置文件保存路径

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    image_train[idx] = image
    mask_train[idx] = mask
  
  return image_train, mask_train

  def testImageFetch(test_id):
  image_test = np.zeros((len(test_id), 101, 101), dtype=np.float32)

  for idx, image_id in tqdm(enumerate(test_id), total=len(test_id)):
    image_path = os.path.join(test_image_dir, image_id+'.png')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255#/255限制输入在0-1
    image_test[idx] = image

  return image_test

  def do_resize2(image, mask, H, W):#图像缩放
  image = cv2.resize(image, dsize=(W,H))
  mask = cv2.resize(mask, dsize=(W,H))
  return image, mask

  def do_center_pad(image, pad_left, pad_right):
  return np.pad(image, (pad_left, pad_right), 'edge')
```
---
* <font color=red>**np.float32**</font><br>是将改变数值数据类型
* <font color=red>**np.zeros（）**</font><br>(shape,dtype=float,order='C')提供给定形状和类型的新数组，并用0填充。<font color=red>生成全零数组</font><br>
**shape**参数用于定义数组的尺寸。<br>
参数用于我们要在其中创建数组的形状, 例如(3, 2)或2。<br>**dtype**参数用于定义数组的所需数据类型。默认情况下, 数据类型为numpy.float64。此参数对于定义不是必需的。<br>
**order**参数用于定义我们要在内存中存储数据的顺序, 即行主要(C样式)或列主要(Fortran样式)
* <font color=red>**os.path.join（）**</font><br>
连接两个或更多的路径名组件
* <font color=red>**tqdm（）**</font><br>
用来显示进度条模块
* <font color=red>**enumerate（）**</font><br>
enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
* <font color=red>**astype（）**</font><br>
转换数据类型
* <font color=red>**cv2.resize（）**</font><br>
缩放图像：cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])<br>

|参数|描述|
|---|---| 
src|【必需】原图像
dsize|【必需】输出图像所需大小
fx|【可选】沿水平轴的比例因子
fy|【可选】沿垂直轴的比例因子
interpolation|【可选】插值方式

* <font color=red>**astype（）**</font><br>
* <font color=red>**astype（）**</font><br>
