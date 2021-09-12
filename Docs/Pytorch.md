# Tensor初始化

## 初始化一个range

**torch.arange(start:int, end:int)**

说明：初始化一个1维的数组





# tensor对象的方法

## 转换形状

**tensor.view(shape:[int, tuple])**

说明：将一个tuple的维度进行转换



# 网络/计算层的初始化

## 最大池化层初始化

**torch.nn.MaxPool2d(size:[int, tuple], padding:[int, tuple], stride:[int, tuple])**

说明：初始化一个最大池化层

参数：

​	size: 池化层形状

​	padding: 输入填充，int类型代表4面填充，(int, int)代表高填充，宽填充

​	stride: 步幅，类型同padding

## 平均池化层初始化

**torch.nn.AvgPool2d(size:[int, tuple], padding:[int, tuple], stride:[int, tuple])**

说明: 初始化一个平均池化层

参数同最大池化层



# 拼接

**torch.cat(tensors:iterable, dim)**

说明：拼接多个张量

参数：

​	tensors: 多个tensor

​	dim: 维度，按照维度进行拼接, dim=0：行拼接， dim=1:列拼接



**torch.stack(tensors:iterable)**

说明：沿着新维度，对输入张量序列进行连接，tensors中的所有张量形状应当相同

简单点说就是：把2个2维张量拼接成一个3维张量