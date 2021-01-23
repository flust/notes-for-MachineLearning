### Optimization for Deep Learning 1-2

---

为了使损失函数尽可能低



1. 两种setting

on-line - one pair of data at a time step

off-line - pour all data into the model at every time step



2. 一些优化损失函数的方法(核心都是梯度下降)

* （1）SGD  （1847年）

  算法步骤如下：

  ​		Start at position $\theta^{0}$

  ​		Compute gradient at $\theta^{0}$

  ​		Move to $\theta^{1} = \theta^{0} - \eta \nabla L(\theta^{0})$

  ​		Compute gradient at $\theta^{1}$

  ​		Move to $\theta^{2} = \theta^{1} - \eta \nabla L(\theta^{1})$

  ​		...

  ​		Stop until $\nabla L(\theta^{t}) \approx 0$

  

* （2）SGD with Momentum(SGDM)   （1986年）

  算法步骤如下

  ​		Start at point $\theta^{0}$

  ​		Movement $v^{0} = 0$

  ​		Compute gradient at $\theta^{0}$

  ​		Movement $v^{1} = \lambda v^{0} - \eta \nabla L(\theta^{0})$

  ​		Move to $\theta^{1} = \theta^{0} + v^{1}$

  ​		Compute gradient at $\theta^{1}$

  ​		Movement $v^{2} = \lambda v^{1} - \eta \nabla L(\theta^{1})$

  ​		Move to $\theta^{2} = \theta^{1} + v^{2}$

  做一个在时间上的累积，就好像过去的时间里给了一个动量/动能，即使遇到比较平的位置，也可以继续移动，一定程度上可以解决局部最优的问题。

  之前的梯度做加权加上当前的梯度



* （3）Adagrad   （2011年）

  算法核心如下：

  ​		$\theta_{t} = \theta_{t-1} - \frac{\eta}{\sqrt{\sum_{i=0}^{t-1}}(g_{i})^2}g_{t-1}$

  如果前几步步长很大，那么分母就会很大，后面的变化会变得很小



* （4）RMSProp   （2013年）

  算法核心如下：

  ​		$\theta_{t} = \theta_{t-1} - \frac{\eta}{\sqrt{v_{t}}}g_{t-1}$

  ​		$v_{1} = g_{0}^2$

  ​		$v_{t} = \alpha v_{t-1} + (1-\alpha)(g_{t-1})^2$

  和Adagrad类似，会更加重视最近的梯度，最近的步长很大的话就会适当缩小，最近的步长很小的话会适当增大



* （5）Adam   （2015年）

  核心：SGDM + RMSProp

  ​		$\theta_{t}=\theta_{t-1} - \frac{\eta}{\sqrt{\hat{v_{t}}}+ \epsilon}\hat{m_{t}}$

  其中：

  ​		$\hat{m_{t}}=\frac{m_{t}}{1-\beta_{1}^{t}}$

  ​		$\hat{v_{t}}=\frac{v_{t}}{1-\beta_{2}^{t}}$

  ​		$\beta_{1} = 0.9$

  ​		$\beta_{2} = 0.999$

  ​		$\epsilon=10^{-8}$



**3. 实际应用**

​	BERT - Adam

​	TRANSFORMERS - Adam

​	Tacotron - Adam

​	YOLO - SGDM

​	Mask R-CNN - SGDM

​	ResNet - SGDM

​	Big-GAN - Adam

​	MAML - Adam



**4.较新工作**

​	Adam比较快，SGDM比较稳

​	SWATS - 2017 - Begin with Adam(fast), end with SGDM

​	AMSGrad - 2018 - improve Adam

​	AdaBound - 2019 - improve Adam

​	Cyclical LR - 2017 - improve SGDM

​	SGDR - 2017 - improve SGDM

​	One-cycle LR - 2017 - improve SGDM







### PyTorch

np.array不会自动做一个graph

torch.tensor会自动做gradient

