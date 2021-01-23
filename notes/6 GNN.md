## Graph Neural Network

数据之间可能会有复杂关系，比如人际关系网络。

#### 1. GNN的应用场景

* Supervised classification - 有监督的分类（图分类，结点分类）
* Semi-Supervised Learning 
* Representation learning: Graph InfoMax  学到一个比较好的feature(node feature / graph feature)
* Generation: GraphVAE, MolGAN

​    例如 某个结构是否属于某个类别 / 判断一个分子结构会不会突变 / 生成出满足某些条件的化学分子，此外，提出几个小问题：

​	数据可能会非常大，20k个节点怎么办？

​	如果不是所有节点都有标签怎么办？

#### 2. Task, Dataset, and Benchmark

* Task

  Semi-supervised node classification

  Regression

  Graph classification

  

#### 3. How？

基本思想是convolution的方式，

* **Spatial-based GNN**（例如GAT）

  1. Terminology (术语)

     用一个 kernel 做 convolution, 从而从 layer i 得到 layer i+1

     * Aggregate: 

     ​	用 neighbor feature 来 update 下一层的 hidden state，简单来说就是用该层该点的邻居的值(经过convolution)来计算下一层该点的值

     * Readout:

     ​	把所有 node 的 feature 集合起来代表整个 graph

  2. 几个例子

     ​	**NN4G**(Neural Networks for Graph) - 将各层所有节点相加作为各层的 feature, 然后各层 feature 加权求和得到整个图的 feature

     ​	**DCNN**(Diffusion-Convolution Neural Network) - 将同一个结点在不同层的表示($ h_{1}^{0},h_{1}^{1}...,h_{1}^{k} $)排成一个矩阵，经过一个transform(乘以一个系数矩阵)之后得到一个结点的 feature

     ​	**MoNet**(Mixture Model Networks) - 定义了结点距离的概念$ u(x,y) = (\frac{1}{\sqrt{deg(x)}},\frac{1}{\sqrt{deg(y)}})^T $

     ​	**GraphSAGE** - 它的 aggregation 用了 mean / max-pooling / LSTM

     ​	**GAT**(Graph Attention Networks) - 对邻居做 attention, 对不同的相邻结点对计算 energy $f(h_{3}^{0}, h_{0}^{0}) = e_{3,0}$, 这个 energy 代表了对于中间结点来说，周围结点有多重要，加权得到下一层的 feature

     ​	**GIN**(Graph Isomorphism Network) - 比较理论，提出有些方法是work的，有些是不work的。提出用sum 代替 mean / max , mean-pooling / max-pooling 会忽视掉图的结构。

     

* **Graph Signal Processing and Spectral-based GNN** （例如GCN）

  将原本的图经过傅立叶变换，卷积核也做傅立叶变换，两者相乘的结果再做傅立叶逆变换

  1. 信号与系统的内容

     在一个n维空间中，每个向量可以写成一组基的线性组合，常用的一组基是 cos 和 sin

     在时域上，

  2. Spectral Graph Theory

     * 基本概念

       图 Graph: G = (V, E), N = |V|

       邻接矩阵 adjacency matrix: $A \in R^{N \times N}$

       度数矩阵 degree matrix: $D \in R^{N \times N}$, 只有对角线记录每个节点的度数

       结点上的signal, signal on graph(vertices): $f: V \rightarrow R^{N}$

     * 图拉普拉斯矩阵 Graph Laplacian

        $L = D - A$, $L$ 是半正定的，对称的

       做谱分解(spectral decomposition)$L = U\Lambda U^{T}$, 其中$ \lambda_{i} $叫做 frequency, $u_{i}$是$\lambda_{i}$对应的basis

       ​	（举个例子？）

       $L$ 可以看作是图上的一种操作 $Lf = (D - A)f = Df - Af$

     * 两个结点 i 和 j的信号能量差

       首先，频率越大，相邻两点之间的信号变化量越大

       $f^{T}Lf = \frac{1}{2}\sum_{v_{i} \in V}\sum_{v_{j} \in V} w_{i,j}(f(v_{i}) - f(v_{j}))^{2}$，$w_{i,j}$是邻接矩阵A中的值

       上式 represents "power" of signal variation between nodes

     * **定义图上的傅立叶变换 (!!!!!!! Spectral 核心？)**

        ​	$\hat{x} = U^{T}x$ 将 vertex domain 上的值转换到 Spectral domain 上
     
        ​	逆变换： $x = U\hat{x}$
     
     * **Spectral Graph Theory**
     
        ​	频域上做卷积 $\hat{y} = g_{\theta}(\Lambda)\hat{x}$ (卷积核与原数据相承)  $g_{\theta}(\Lambda)$ 叫做 filter
     
        ​	再转换回时域 $y = U \hat{y} = U g_{\theta}(\Lambda)U^{T}x$
     
        ​	**需要学习的就是 filter（一个关于拉普拉斯矩阵的函数）**，$g()$可以是任何函数
     
  3. 几个例子
  
     * ChebNet
  
       ​	$g_{\theta}(L) = \sum_{k = 0}^{K}\theta_{k}L^{k}$    	$g_{\theta}(\Lambda) = \sum_{k = 0}^{K}\theta_{k}\Lambda^{k}$    
  
       使用L的k次多项式作为filter，K-localized，但会出现时间复杂度过高的问题
  
       可以使用切比雪夫多项式  	$g_{\theta^{'}}(\hat{\Lambda}) = \sum_{k = 0}^{K}\theta_{k}^{'}T_{k}(\hat{\Lambda})$
  
       使用多个filter计算多个结果$y_{1},y_{2},y_{3}...$
  
     * GCN
  
       ​	在 ChebNet 的形式上，K取1，公式如下
  
       ​		$y = g_{\theta^{'}}(L)x = \theta_{0}^{'}x + \theta_{1}^{'}\hat{L}x = \theta(I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x$
  
       renormalization trick: $I_{N} + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \rightarrow \hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$
  
       ​	$H^{l + 1} = \sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$

#### 4. Benchmarks tasks

* Graph Classification: SuperPixel MNIST and CIFAR10
* Regression: ZINC molecule graphs dataset
* Node classification: Stochastic Block Model dataset
* Edge classification: Traveling Salesman Problem





相关库 Deep Graph Library

