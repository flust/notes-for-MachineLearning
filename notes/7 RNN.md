## Recurrent Neural Network(RNN)

#### 1. 基本概念

* 应用：slot filling 完形填空

  例如：订票系统中:

  ​		I would like to arrive Taipei on November 2nd

  ​		slot: Destination(Taipei) / time of arrival(November 2nd)

  ​	判断某个词是不是某个slot，直接以单词为输入进行全连接，无法处理上下文信息

* RNN

  RNN - 让Neural network有记忆力

  The output of hidden layer are stored in the memory

  设置memory单元，每次前一个时间的内容有一部分存在 memory 单元里

  Memory can be considered as another input

* 简单的例子：

  ​    例子：三层，每层两个结点，全连接，weights全是1，激活函数全是Linear，没有bias，记忆单元同样连接隐藏层的输入

  ​	input sequence: [1,1],[1,1],[2,2]

  ​	memory init: [0], [0]

  首先输入[1,1]，隐藏层计算结果为[2,2]，output sequence[4,4]，此时 memory 变为[2],[2]

  然后再输入[1,1]，隐藏层计算结果为[6,6]，output sequence[12,12]，此时 memory 变为[6],[6]

  再输入[2,2]，隐藏层计算结果为[16,16]，output sequence[32,32]，此时 memory 变为[16],[16]

  input sequence 的顺序会影响结果

![image-20201217152824831](/Users/yangchen13/Library/Application Support/typora-user-images/image-20201217152824831.png)

​	例子： arrive Taipei on November 2nd

​	首先输入 arrive，输出一个 y1， memory 里存 arrive

​	然后输入 Taipei，根据 arrive(memory) 和 Taipei 作为输入计算得到 y2

​	这样输入同样的词汇，也可以根据上一个词得到不同的预测值

​	架构可以任意设计

​	比如多个hidden layer



#### 2.多个变形

* Elman Network: 上述结构，存储hidden layer的值

* Jordan Network: 存的是output的值

* Bidirectional RNN：双向RNN，同时train一个正向的RNN和一个逆向的RNN，两个hidden layer拿出来接到一个output layer上



#### 3. Long Short-term Memory(LSTM)

长短期记忆网络，注意是 比较长的短期记忆网络

* 基本架构

  基本单元：三个gate(Input Gate, Output Gate, Forget Gate)，一个Memory Cell

  ​	当外界想写入Memory Cell 时，需要通过一个Input Gate，当Input Gate打开时才能写入，表示输入数据的重要程度，输出的地方需要通过一个Output Gate，Forget Gate 表示原有内容记住的程度 ，什么时候需要把过去记得的东西忘掉，三个Gate都通过学习获得。

  ​	其参数数量为一般的 Neural Network 的四倍。

* 公式表示

  $z_{i},z_{o},z_{f}$表示控制三个门的信号，activation function 一般都是 sigmoid

  Memory里的值更新方程为：$c^{'} = g(z)f(z_{i}) + cf(z_{f})$

  *(更新值) = (输入值) x (input gate打开的程度) + (memory里原有内容) x (forget gate忘记的程度)*

  然后 output 为 更新的值 再乘以 output gate 的值

  <img src="/Users/yangchen13/Library/Application Support/typora-user-images/image-20201217153717230.png" alt="image-20201217153717230" style="zoom:50%;" />

* 矩阵表示

  ​	多个memory的值拼接起来得到一个向量 $c^{t - 1}$

  ​	在时间点t，input的vector $x^{t}$，乘上一个 matrix 得到一个 z，z的每一个dimension表示操控每个memory cell的输入信号，$x^{t}$乘上四个个系数矩阵得到四个 vector $z^{f}, z^{i}, z^{o}, z$ 去操控所有 memory cell 的运作

  ​	所有的 cell 可以一起运算，变成矩阵运算 

  ​		一般，输入的x由几个数据拼接起来(peephole) $[c^{t - 1}, h^{t - 1}, x^{t}]$

  Keras supports:  LSTM GRU SimpleRNN (layers)

<img src="/Users/yangchen13/Library/Application Support/typora-user-images/image-20201217154030531.png" alt="image-20201217154030531" style="zoom:50%;" />



#### 4.loss

training sentences: 

​	原句 arrive Taipei on November 2nd

​	标签 other dest other time time



训练方法 Backpropagation through time(BPTT) 

The error surface is rough 跳跃性很大

clipping：设置gradient最大值阈值

Trimming: 把重复的东西去掉



解决梯度消失：LSTM可以解决 gradient vanishing 的问题

​	Memory 和 input 是相加而不是直接替换

​	一旦对memory造成影响，将永远保留

​	

GRU：两个Gate，simpler than LSTM

Clockwise RNN

SCRN



部分应用

* Input and output are both sequences with the same length

* Input is a vector sequence, but output is only one vector

  ​	Sentiment Analysis

  ​	Key Term Extraction

* Both input and output are both sequences, but the output is shorter

  ​	Speech Recognition

  ​		Connectionist Temporal Classification (CTC)

* Both input and output are both sequences with different length -> Sequence to sequence learning

  ​	Machine Translation

  ​		直接把英文的声音讯号转换成中文文字？？

  ​	Syntactic parsing



Sequence-to-sequence Auto-encoder - Text / Speech



RNN作为Encoder，存在memory里面的内容就是想要的

同样用RNN作为Decoder，让他解码之后的结果 x 与 y 越接近越好

Encoder和Decoder是一起训练的



Demo: Chat-bot

​	LSTM Encoder / LSTM Decoder

​	数据样例:  input: How are you, output: I am fine



#### 5. Attention-based Model (RNN 进阶？)

Read Comprehension

​	Query -> DNN / RNN -> Reading Head Controller -> 读取information -> Answer

Visual Question Answering

​	图 -> 回答问题

​	Query -> DNN / RNN -> Reading Head Controller -> 读取图片的位置 -> Answer

Speech Question Answering



#### 6. RNN vs Structured Learning

HMM CRF 部分暂时跳过