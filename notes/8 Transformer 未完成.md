## Transformer

Seq2seq model with "self-attention"



大量用到了 self-attention



#### 1. self-attention layer



self-attention 取代 RNN

​	输入一个 seq 输出一个 seq

​	跟 BiRNN 有同样的能力

​	

input x1-x4   通过一个embedding  变成 a1-a4

接下来进入self-attention layer， 每个a乘上三个不同的transformation得到三个不同的vector，分别命名为 q k v

q代表 query, k代表 key,  v代表 value

接下来 拿每个 query q 去对每一个 key k 做 attention

attention: 吃两个向量，output一个分数，表示两个向量的相关程度

q1对k1做attention $\alpha_{1,1}$， 同理得到$\alpha_{1,2}, \alpha_{1,3}$，具体计算方法：1. Scaled Dot-Product Attention $\alpha_{1,i}=q^{1}*k^{i}/\sqrt{d}$，其中，乘积为 inner product，d是 q和k 的维数  

通过一个softmax 得到$\hat{a}_{1,1}...\hat{a}_{1,4}$

​	$\alpha$就是attention

接下来：$b^{1} = \sum_{i}\hat{\alpha}_{1,i}v^{i}$  即上面得到的结果去乘以value

​	所以b1中包含了 a1-a4的信息



变成矩阵形式：计算QKV$Q = W^{q} I$

attention的matrix: $A = K^{T} Q$

$B = V \hat{A}$



















nohup python run_recbole.py --model=xgboost --xgb_num_boost_round=5000 --dataset=ml-1m --train_at_once=True --print_data=True --save_data=False --print_pred=True >1218log.log 2>&1

