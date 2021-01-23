# Deep Reinforcement Learning

强化学习与有监督学习、无监督学习类似，也是机器学习的一个分支，alpha-go 就是强化学习

有人说 Deep Reinforcement Learning: AI = RL + DL



### 1. 基础内容

基本的强化学习包含的内容：

* Agent
* Environment
* Obervation(也叫State): Environment->Agent, Agent 对 Environment 的观察
* Action: Agent->Environment, Machine 做的事情，会影响环境
* Reward: Environment->Agent, 环境给出 Action 对环境的影响的反馈，Agent 根据 Reward 调整 model

比如 AlphaGo:

	* Agent: 自己
	* Environment: 对手
	* Observation: 棋盘，对手落子
	* Action: 落子在棋盘上

简单来说，整个过程就是，机器观测环境，根据模型作出行为，然后环境给予机器反馈，机器根据反馈修改模型。



### 2. 有监督学习 vs 强化学习

* 有监督学习：

  看到一个状态，得出下一步棋下在哪

* 强化学习

  ​	找一个人下围棋，如果赢了就得到 positive reward，如果输了就得到negtive reward，要自己想办法得出怎么样下是好的

  ​	需要大量的训练，也可以两个Agents互相训练

例如，训练一个聊天机器人：

* 有监督学习：

  Hello -> Hi

  Bye bye -> Good bye

* 强化学习

  ​	Agent 胡乱说话，根据对方的 Reward 来调整自己的model，比如对方是否生气

  ​	也可以让两个 agents 互相对话，但是也需要有人告诉他们讲的好不好

更多的一些应用: Interactive retrieval / Flying Helicopter / Driving / Text generation

​	下面是机器学电玩，有两个特点 1.observe就是画面  2.行为是自己学出来的

​	Gym: https://gym.openai.com

​	Universe: https://openai.com/blog/universe



### 3. 强化学习的难点

1. Reward 的出现会有延迟，可能多步以后才会有 reward
2. Agent 的行为会影响它接收到的东西，它需要会探索这个世界(Exploration)



### 4. Outline

* Policy-based: Learning an Actor

* Value-based: Learning a Critic

* Actor + Critic: 最强的方法: Asynchronous Advantage Actor-Critic(A3C)

* Alpha Go: policy-based + value-based + model-based



### 5. Policy-based Approach: Learning an Actor

Machine Learning = Looking for a Function

在强化学习中，Observation就是函数输入，Action就是函数输出，利用Reward去找最好的Function（ Actor / Policy = $\pi$(Observation) ）（Policy指的就是那个Actor / Agent（可以是一个Neural Network））

深度学习三个步骤：

 1. Neural network as Actor

    输入: observation as a vector or a matrix

    输出: action corresponds to a neuron in output layer

    ​	根据输出概率 stochastic 随机选择行为

    

 2. goodness of function

    在监督学习中，根据 loss 判断一个 function 的好坏

    * Given an actor $\pi_{\theta}(s)$ with network parameter $\theta$
    * Use the actor $\pi_{\theta}(s)$ to play the video game
      * Total reward: $R_\theta = \sum^{T}_{t = 1}r_{t}$ , Even with the same actor, $R_{\theta}$ is different each time (因为存在随机性，所以每次结果会不一样，这是一个随机变量)
      * 所以，最终目标是最大化 $\hat{R_{\theta}}$ 的期望值
    * An episode is considered as a trajectory $\tau$
      * $\tau = \{s_{1}, a_{1}, r_{1}, ... , s_{T}, a_{T}, r_{t}\}$
      * $R(\tau) = \sum^{N}_{n = 1}r_{n}$
      * $\tau$ 的概率是可以计算的
      * $\hat{R_{\theta}} = \sum_{\tau} R(\tau)P(\tau|\theta)$, 但是，得到所有的是不可能的
      * 所以，Use $\pi_{\theta}$ to play the game N times, obtain ${\tau^{1}, \tau^{2},..., \tau^{N}}$ 相当于 Sampling $\tau$ from $P(\tau|\theta)$ N times.
      * 所以，$\hat{R_\theta} \approx \frac{1}{N}\sum^{N}_{n = 1}R(\tau^{n})$

 3. pick the best function

    Gradient Ascent 梯度上升

    原问题：$\theta^{*} = arg max(\hat{R_{\theta}}),\hat{R_{\theta} = \sum R_{\theta}P(\tau|\theta)}$ , 对 $\theta$ 做梯度上升

    （*数学公式略，需要时见网课，有空推导一下*）



// 网课 1 60:00

