![[Pasted image 20260405101427.png]]
# 目标检测
## 任务
输入一张RGB图像
输出一组检测到的目标，每个目标都有一个类标签和一个框
## 挑战
1. 多个输出（多个目标）
2. 多种输出（每个目标要输出类别和框）
3. 更大的输入图像
## 工作
### 检测一个目标
CNN永远可以理解为一个特征提取器（KNN都能用他做一些很棒的事情），拥有了图片特征了以后，我们就要考虑两个输出了——What&Where，分别用两个全连接神经网络解决
![[Pasted image 20260405102437.png]]CNN+两个并行的全连接神经网络会产生两个loss，解决方法就是直接相加
但是我们实际上在一张照片里会有多个目标，对于Where来说，我们会输出多个框的坐标
### 检测多个目标
#### 滑动窗口
我们的CNN对子区域识别——是否是背景？是否是猫？是否是狗？
然后我们滑动这个子区域，遍历每一个位置
显然这不行，我们不仅会重复识别同一个目标，而且会导致极大量的计算
#### 区域提议
我们使用一种外部算法，来提议一组图像的候选区域（选择高概率可能是目标的区域），算法一般是 **“选择性搜索”**（运行在CPU上的算法，可以给约2000个待选区域）
![[Pasted image 20260405104659.png]]
#### R-CNN: Region-Based CNN(*slow*)
对于一个图像，先运行选择性搜索，它会给我们多个**不同大小**的区域图片，然后我们把他强行变形(224x224)为固定尺寸，输入CNN
CNN会输出Class分数以及框的变形参数（我们在提议的区域基础上做一些修改）
![[Pasted image 20260405105624.png]]
我们一般使用如下参数化
$$\mathrm{Region\ proposal:}\ (p_x,p_y,p_H,p_W)$$
$$\mathrm{Transform:}\ (t_x,t_y,t_H,t_W)$$
$$b_x=p_x+p_Wt_x\quad b_y=p_y+p_Ht_y$$
$$b_W=p_W\exp(t_W)\quad b_H=p_H\exp(t_H)$$
注意到，我们并不需要输入位置信息，同样的，我们输出的变换信息也只是改变大小

也可以不用添加Background的类标签，只需要让大于某个阈值的框输出即可对应的类即可，小于则不输出(Background)
#### 比较框(IoU)
![[Pasted image 20260405114253.png]]
IoU=两个集合的交集比并集
IoU大于0.5可以认为不错，大于0.7就是很不错，大于0.9就是几乎完美
#### Non-Max Suppression(NMS)
![[Pasted image 20260405114655.png|462]]
选择得分最高的框（这里是蓝色的），然后计算和每一个框的IoU，然后**删除**大于某个阈值(0.7)的IoU对应的那个框
![[Pasted image 20260405114908.png|148]]
然后不断循环操作，直到没有超过阈值的IoU
NMS对这种图像就会失效：
![[Pasted image 20260405115147.png]]
#### 整体的评估指标(mAP)
先对图像运行目标检测器，然后用NMS去重
对于**每一个类别**，我们计算平均精度(AP)
例如我们有5个框，每个框都有一个狗类的评分（蓝色方块），下面的三个橙色框代表的是真实标注狗的位置
![[Pasted image 20260405143622.png|387]]
然后我们把每个蓝块和橙块连接(IoU>0.5)，计算：
1. Precision：匹配的蓝块的数量/选中的蓝块数量
2. Recall：被连接的橙块数量（连接后）/选中的蓝块数量
然后再Recall-Precision坐标系上画出点
![[Pasted image 20260405144323.png|300]]
![[Pasted image 20260405144442.png|331]]
我们连接每一个点，然后计算积分，就得到了Dog的AP，计算每个类的AP然后取算术平均就能得到mAP
AP的意义很好理解——模型给的高概率是否准确
实际测试中，我们取不同的IoU阈值然后计算mAP
#### Fast R-CNN
R-CNN非常昂贵（一次要运行2000次CNN的前向传播）
解决方法很简单，由于选择性搜索有大量重叠的提议，所以我们可以**共享计算**，减少CNN不必要的运算量。为此，我们对每个提议运行CNN，然后在固定提议的大小
![[Pasted image 20260405151504.png]]

我们先用一个Backbone(ResNet, AlexNet...)网络来把图片转化为Feature Map，然后把每个proposal映射到Feature Map上，然后对每个proposal进行PRN(Pre-Region Network，大概是三层)，虽然我们仍然进行2000次运算，但是这比原来的CNN计算量小很多

这又引入了一个问题，我们在神经网络传播过程中进行操作，这必须要求操作是可微分的（否则无法学习）
##### RoI Pool(region of interest pooling)
Resize方法（不通过扭曲Feature Map，而是通过Max Pooling）
注意我们的Proposal的边框映射以后不一定能和每个像素对齐，我们先移动，使之对齐
![[Pasted image 20260405153249.png]]
然后我们将这个区域化为2x2的区域（尽量），然后将这个2x2（或2x3）区域，对每个通道层进行最大池化，就能得到固定大小
![[Pasted image 20260405153609.png]]
#### Faster R-CNN
不使用选择性搜索算法，而是使用 **区域提议网络(RPN)** 做这件事
![[Pasted image 20260405154130.png]]
##### RPN
由于是线性映射，在Feature Map的每一个点都对应输入图的一个点
![[Pasted image 20260405154642.png|505]]我们想象一个anchor box，他的中心位于某个点
我们的RPN就是一个二分类网络，我们**人为规定**一个anchor box的大小，然后RPN就会决定这个anchor box里有没有目标（绿色有红色没有）
![[Pasted image 20260405155447.png]]和R-CNN类似，RPN还会输出一个变换系数，把我们的anchor box变得更合适，此处我们把绿框变成黄框
![[Pasted image 20260405155750.png|295]]
在实践中，只有一个固定大小的Anchor通常不够，我们用各种大小形状的Anchor
![[Pasted image 20260405155907.png|231]]
我们需要训练4种损失：
1. RPN 分类：anchor box里有没有Object
2. RPN 回归：变形系数
3. Object 分类：分类每个Proposal是背景还是Object
4. Object 回归：变形系数
##### 总结
先直接把图片输入Backbone CNN，得到Feature Map
然后生成Anchor，利用GT（人工标注的框）给Anchor打标签（利用IoU）
过RPN(Region Proposal Network)，生成提议
然后RoI pooling
最后我们的每个Detection Head（CNN）分类并且生成更精确的框