## Gradient Descent

Review： 需要最小化 loss function， 提出梯度下降法

**1. 损失函数 loss function** 

​	判断一个预测函数 function 好坏的函数（想怎么定义怎么定义），比如；将一个预测函数的预测值和真实值取差值然后求平方和。其大小表示了一个预测函数 function 的好坏。

​	常用的（logLoss, coss-entropy loss, quadratic loss）

**2. 基本定义**

​	假设 function 包括两个参数 $\{\theta_{1}, \theta_{2}\}$,  目标是找到最好的两个参数使得损失值越小越好（使得预测函数越好越好）， 具体操作步骤如下：

 * 随机选择初始值 $ \theta^{0} = \left[ \begin{matrix} \theta_{1}^{0} \\ \theta_{2}^{0} \end{matrix}  \right] $
 * $\left[\begin{matrix}\theta_{1}^{1}\\\theta_{2}^{1}\end{matrix} 
   \right] = \left[\begin{matrix}\theta_{1}^{0}\\\theta_{2}^{0}\end{matrix} 
   \right] - \eta \left[\begin{matrix}\partial L(\theta_{1}^{0}) / \partial \theta_{1}\\\partial L(\theta_{2}^{0}) / \partial \theta_{2}
   \end{matrix} \right]$

​    其中，$\eta$是学习率(学习的速度)，后面的那个$\left[\begin{matrix}\partial L(\theta_{1}^{0}) / \partial \theta_{1}\\\partial L(\theta_{2}^{0}) / \partial \theta_{2}
\end{matrix} \right]$就叫做梯度 gradient  $\nabla L(\theta)$ .至于为什么这么做，先不用管（数学证明在后面）。

​    不断重复上述过程，可以得到越来越好的参数。

​	另： Gradient 是 loss 的等高线法线方向（数学上梯度的解释）

**3. 学习率的解释**

​	学习率 Learning Rate 表示每次迭代更改的步长，一般来说，随着迭代进行，loss会越来越小。如果学习率过大，可能会直接越过山谷（Loss小的地方）。

​	所以需要合理的设置学习率：

* **Adaptive Learning Rates （随迭代修改学习率）**

  ​    这种方法是随着迭代进行，逐步减小学习率，一开始离目标很远的时候，步长设置较大，在快接近目标时，设置步长较小，避免错过最低点。

  * Adagrad（ 一种比较简单的方法）

    ​    每个参数的学习率每次变成之前的除以$\sqrt{t + 1}$，再除以它之前的值的 root mean square （平方和取平均再开根号）

    $w^{t + 1} \leftarrow w^{t} - \frac{\eta ^{t}}{\sigma^{t}} g^{t}$

    其中: $\sigma^{n} = \sqrt{\frac{1}{t + 1}\sum_{0}^{t}(g^{i})^{2}}$ (root mean square),   $ \eta ^ t = \frac{\eta}{\sqrt{t + 1}}$

    化简之后: $w^{t + 1} \leftarrow w^{t} - \frac{\eta}{\sqrt{\sum_{0}^{t}(g^{i})^{2}}}g^{t}$

* **Second derivative（考虑二阶导数）**

  ​    设置步长为 $\frac{|first\ derivative|}{second\ derivative}$

* **Stochastic Gradient Descent（随机梯度下降）**

  ​	随机选择样本，只考虑样本进行梯度下降，多次重复此过程，可以降低计算复杂度。

  ​	比如，整体做梯度下降的时间消耗可能和随机梯度下降几十次的时间相同，也有可能整体梯度下降计算时间无法接受。每次用样本进行参数更新，足够多次后效果类似（可能更好）。

* **Feature Scaling（对特征做标准化）**

  ​	多个特征的情况下，不同特征的数量级可能不一样且对结果影响比较大，可以对特征进行标准化，减去平均值mean，除以标准差standard deviation： $x \leftarrow \frac{x - \hat{x}}{\sigma}$

**4. 数学解释**

​	给出一个点，可以找到该点附近一个小范围内的最低点（也就是迭代更新的步骤），那么问题就变成了给一个点，找附近小范围的最低点。

​		泰勒级数（泰勒展开）:

​		    $h(x) = \sum_{k = 0}^{\infin}\frac{h^{(k)}(x_{0})}{k!}(x - x_{0})^k,\  (x_{0}处的值不可计算)$

​		展开之后:

​			$h(x) = h(x_{0}) + h^{'}(x_{0})(x - x_{0}) + \frac{h^{''}(x_{0})}{2!}(x -x_{0})^{2} + ...$

​		当x接近$x_{0}$时: 

​			$h(x) \approx h(x_{0}) + h^{'}(x_{0})(x - x_{0})$

​		多元泰勒级数同理:

​			$h(x,y) = h(x_{0},y_{0}) + \frac{\partial h(x_{0},y_{0})}{\partial x}(x - x_{0}) + \frac{\partial h(x_{0},y_{0})}{\partial y}(y - y_{0}) + ...$

​			$h(x,y) \approx h(x_{0},y_{0}) + \frac{\partial h(x_{0},y_{0})}{\partial x}(x - x_{0}) + \frac{\partial h(x_{0},y_{0})}{\partial y}(y - y_{0})$

​	基于泰勒级数，对损失函数做展开：

​		$L(\theta) \approx L(a,b) + \frac{\partial L(a,b)}{\partial \theta_{1}}(\theta_{1} - a) + \frac{\partial h(a,b)}{\partial \theta_{2}}(\theta_{2} - b)$

​	令 $s = L(a,b),\ u = \frac{\part L(a,b)}{\part \theta_{1}},\ v = \frac{\part L(a,b)}{\part \theta_{2}}$, 其可视作常数

​		$L(\theta) \approx s +u(\theta_{1} - a) + v(\theta_{2} - b)$

​	问题变成了，在一个小范围内（$(\theta_{1} - a)^2 + (\theta_{2} - b)^2 \le d^2$），找$\theta_{1}, \theta_{2}$使得$L(\theta)$最小

​	$L(\theta)$可以看成两个向量$(u,v)和(\Delta \theta_{1}, \Delta \theta_{2})$的乘积，当两向量刚好反向时，损失函数最小，也就是:

​		$\left[ \begin{matrix}\Delta\theta_{1}\\ \Delta\theta_{2} \end{matrix} \right] = - \eta \left[ \begin{matrix}u\\ v \end{matrix} \right]$,	$\left[ \begin{matrix}\theta_{1}\\ \theta_{2} \end{matrix} \right] =\left[ \begin{matrix}a\\ b \end{matrix} \right] - \eta \left[ \begin{matrix}u\\ v \end{matrix} \right]$

**5. 一些问题和局限性**

* 有些位置导数为零但并不是极值点

* 有些位置比较平缓导致步长很小
* 受限于局部最优值



​		