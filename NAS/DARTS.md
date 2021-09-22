### DARTS

**Paper: [DARTS: Differentiable Architecture Search[C]. ICLR, 2019](https://arxiv.org/abs/1806.09055)**

#### Abstract

第一种基于梯度下降的NAS方法。

#### 1. Introduction

现有最好的架构搜索算法在计算上要求很高，尽管它们的性能非常出色。目前已经提出了几种加速的方法，本文从一个不同的角度来处理这个问题，并提出了一种高效的架构搜索方法，称为DARTS (Differentiable architecture search)。

**Code：https://github.com/quark0/darts**

#### 2.  Differentiable Architecture Search

在本文的搜索空间中，架构或其中单元的计算过程表示为有向无环图，然后本文为搜索空间引入了一个简单的连续松弛方案，为网络结构及其权重的联合优化提供了一个可微的学习目标，最后本文提出了一种近似方法，使算法在计算上可行且高效。

##### 2.1  **Search Space**

本文寻找一个计算单元作为最终架构的block，所学习的单元可以堆叠形成卷积网络，也可以递归连接形成递归网络。

单元是由N个节点的有序序列组成的有向无环图。每个节点*x*(*i*)是一个隐含的表征（例如卷积网络中的feature map），并且每个有向边(*i*, *j*)与变换*x*(*i*)的某个操作*o*(*i*, *j*)相关联。假设某单元有两个输入节点和一个输出节点。对于卷积单元，输入节点定义为前两层单元的输出。对于循环单元，输入节点定义为当前步骤的输入和前一步骤的状态。单元的输出通过对所有中间节点应用缩减操作（例如串联）来获得。

每个中间节点都基于其所有前置节点进行计算：
$$
x^{(j)}=\sum_{i<j} o^{(i, j)}\left(x^{(i)}\right)
$$
在操作中存在一种特殊的零操作，以指示两个节点之间缺少连接。因此，学习单元的任务简化为学习其边缘的操作。

##### 2.2 Continuous Relaxation and Optimization

设 $O$ 是一组候选操作（例如：卷积、最大池、零），其中每个操作表示要应用于的 $x^{(i)}$ 某个函数 $o(\cdot)$ 。为了使搜索空间连续，我们将特定操作的分类选择松弛为所有可能操作的一个softmax操作：
$$
\bar{o}^{(i, j)}(x)=\sum_{o \in \mathcal{O}} \frac{\exp \left(\alpha_{o}^{(i, j)}\right)}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left(\alpha_{o^{\prime}}^{(i, j)}\right)} o(x)
$$
