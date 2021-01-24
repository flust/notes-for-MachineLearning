## Transformer - Seq2seq model with "self-attention"

特殊点在于大量用到了 self-attention

BERT 就是无监督的 Transformer

### 1. Sequence

最常用的是RNN，输入是一串vector seq，输出是另一串vector seq

​	如果是单向RNN，输出时已经把前面的都看过

​	如果是双向RNN，输出时已经把整个句子都看过

但是RNN不容易平行化，也可以用CNN做 seq2seq，但CNN只能考虑有限的内容，叠好多层才能看到长距离资讯



### 2. Self-Attention

输入输出与RNN一样，与BiRNN有一样的能力，用Self-Attention代替RNN

1. input $x_{1}-x_{4}$  通过一个 embedding  变成 $a_{1}-a_{4}$

2. 每个a乘上三个不同的 transformation 

   $q^{i} = W^{q}a^{i}$，q代表 query， to match others

   $k^{i} = W^{k}a^{i}$，k代表 key，to be matched

   $v^{i} = W^{v}a^{i}$，v代表 value，information to be extracted

   得到向量$(q_{1},k_{1},v_{1})-(q_{4},k_{4},v_{4})$

3. 用每个 q 对每一个 k 做 attention 得到一个值

   $q^{1},k^{1} \rightarrow \alpha_{1,1}$ ，同理 $\alpha_{1,2},\alpha_{1,3}$

   例如 Scaled Dot-Product Attention: $\alpha_{1,i} = q^{1} *k^{i} / \sqrt{d}$ ，先做内积，d 是 q 和 k 的 dim

4. 将所有的 $\alpha$ 经过一个 softmax 得到 $\hat{\alpha}$

5. 如下计算 b

   $b^{1} = \sum_{i}\hat{\alpha}_{1,i}v^{i}$ ，即上面得到的结果去乘以value，可以认为 $b^{1}$ 是 $v^{1}-v^{4}$ 做 weighted sum 得到的，$b^{1}$ 中包含了 $a^{1}-a^{4}$ 的信息

矩阵形式如下：

1. 首先，计算 QKV

​	$Q = W^{q} I$

​	$K = W^{k} I$

​	$V = W^{v} I$

2. 计算 attention

$$\left[ \begin{matrix} \alpha_{1,1} & \alpha_{2,1} &\alpha_{3,1} &\alpha_{4,1} \\ \alpha_{1,2} & \alpha_{2,2} &\alpha_{3,2} &\alpha_{4,2} \\ \alpha_{1,3} & \alpha_{2,3} &\alpha_{3,3} &\alpha_{4,3} \\ \alpha_{1,4} & \alpha_{2,4} &\alpha_{3,4} &\alpha_{4,4} \\ \end{matrix} \right] = \left[ \begin{matrix} k^{1} \\ k^{2} \\ k^{3} \\ k^{4} \end{matrix}\right] \left[ \begin{matrix} q^{1} & q^{2} & q^{3} & q^{4} \end{matrix}\right]$$

​	 $A = K^{T} Q$  

​	然后做一下softmax

3. 计算 B

$$\left[ \begin{matrix} b^{1} & b^{2} & b^{3} & b^{4} \end{matrix}\right] = \left[ \begin{matrix} v^{1} & v^{2} & v^{3} & v^{4} \end{matrix}\right] \left[ \begin{matrix} \alpha_{1,1} & \alpha_{2,1} &\alpha_{3,1} &\alpha_{4,1} \\ \alpha_{1,2} & \alpha_{2,2} &\alpha_{3,2} &\alpha_{4,2} \\ \alpha_{1,3} & \alpha_{2,3} &\alpha_{3,3} &\alpha_{4,3} \\ \alpha_{1,4} & \alpha_{2,4} &\alpha_{3,4} &\alpha_{4,4} \\ \end{matrix} \right]$$

​	$B = V \hat{A}$



### 3. Multi-head Self-attention

以 2 heads 为例：

​	$q^{i,1} = W^{q,1}q^{i}$

​	$q^{i,2} = W^{q,2}q^{i}$

​	同理 k v 做同样的操作，最后计算出 $b^{i,1},b^{i,2}$

​	然后直接拼接起来作为  $b^{i}$ 



### 4. Positional Encoding

在self-attention里没有位置信息

一种方式：将 $a^{i}$ 加上一个位置编码vector $e^{i}$ ，直接相加而不是拼接



### 5. Seq2seq with Attention

​	Encoder 和 Decoder 都用 self-attention 层做替换



### 6. Transformer

44:08