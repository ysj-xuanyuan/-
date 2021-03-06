@[pytoch学习笔记](ysj)

#DataLoader：
参考博客：https://www.cnblogs.com/ranjiewen/p/10128046.html

#python中的iter()函数与next()函数

list、tuple等都是可迭代对象，我们可以通过iter()函数获取这些可迭代对象的迭代器。然后我们可以对获取到的迭代器不断使next()函数来获取下条数据。
反复访问迭代器iter，反复使用next的方法

参考博客：https://blog.csdn.net/caomin1hao/article/details/109494832

#dataloader_PyTorch常见的坑汇总
参考博客：https://blog.csdn.net/weixin_39973009/article/details/111173954

#2D卷积和3D卷积

参考博客：https://zhuanlan.zhihu.com/p/55567098

最显著的区别：对于一张RGB多通道的图像，2D卷积对齐进行多通道的卷积操作，但是在3D卷积中将其视为单通道的卷积操作。对于多张RGB图像的输入，2D卷积只能先多张图像concat在一起再放入网络中做多通道的卷积，这将会模糊图像的独立性，而3D卷积可以保留每张图像的独立性，对多张输入图像做多通道的3D卷积，所以可以提取图像之间的信息。

#补偿lr设置

```python
#源码
torch.optim.lr_sheduler.ReduceLROnPlateau(optimizer,model='min',factor=0.1,patience=10,verbose=False,threshold=0.0001,threshold_model='Rel',coldown=0,min_lr=0,eps=1e-08)

#调用
scheduler = ReduceLROnPlateau(optimizer,'min')
scheduler.step(loss_val)#每调用一次监听一次loss，如果符合提前设置的规则就降低loss

scheduler = StepLR(optimizer,step_size=30,gamma=0.1)#lr每隔30个epoch降低0.1
scheduler.step(loss_val)

```

#训练技巧（防止过拟合等）
##Early Stopping
注意训练loss与测试loss是否一致

用Validation set 做一个选择，在最高点停止（依靠经验）

##Dropout
提升鲁棒性（Robustness），减少over feating.
