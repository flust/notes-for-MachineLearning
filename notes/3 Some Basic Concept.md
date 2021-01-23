# Some Basic Concept

### 1. error 来源

* Estimator 估测值

  $\hat{y} = \hat{f}()$

  $f^{*}$ is an estimator of $\hat{f}$

  $f^{*}$ 与 $\hat{f}$ 的距离为 Bias + Variance

* Bias and Variance of Estimator

  e.g. Estimate the mean of a variable x

  ​	采样 N 个点，然后取平均值，这个结果是 unbiased 

  但是预测 variance 方差时，这种方法是有偏的，是一个Biased estimator

  ​	$E[s^{2}] = \frac{N - 1}{N}\sigma^{2}$

* ERROR 取决于两个地方

  以打靶为例，两个地方为 **瞄准的地方 - Bias** 和 **方差 - Variance**

### 2. 结果分析

* Parallel Universes 平行宇宙 - 多次实验

  多次实验，画出不同结果组合成的图，重复度越高，表示variance越小

  模型越简单，variance越小

* Bias & Variance

  large bias：预测值的期望离靶心比较远，函数次数很低时，large bias

  small bias：预测值的期望离靶心比较近，函数次数很高时，small bias，因为平均起来期望值很接近

  简单的model：large bias & small variance

  复杂的model：small bias & large variance

* overfitting & underfitting

  overfitting：error来自于variance

  underfitting：error来自于bias

通过测试集可以判断error



### 3. 解决

* for bias: 

  redesign your model，add features， more complex model

* for variance: 

  more data, Regularization



### 4. 比赛注意事项

should not do: 只选择公开测试集上最好的model，但隐藏测试集可能不好

Homework: Training Set, Testing Set(public), Testing Set(private)

构建模型的时候不要使用任何测试集信息