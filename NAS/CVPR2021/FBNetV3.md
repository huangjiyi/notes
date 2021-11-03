### FBNetV3

**Paper: [FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining. CVPR, 2021.](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.html)**

#### Abstract

之前的 NAS 方法需要在一系列训练超参数 (也即训练 recipe) 下搜索架构，而忽视了架构-超参数的组合，针对这个问题，本文提出了 Neural Architecture-Recipe Search (NARS)，同时搜索架构和相应的训练超参数。

#### 1. Introduction

本文认为当前已有的 NAS 方法有以下几个缺点：

- **忽视训练超参数**：即只搜索网络架构而不关联训练超参数，然而不同训练超参数会影响结果。
- **只支持一次性使用**：许多卷积模型搜索方法每次只能在一组特定的约束条件下生成一个模型，这意味着对于在不同的约束条件下，每次都需要重新进行一次 NAS 方法，或者使用人工的手段对搜索到的模型进行缩放，但是这不是最优的。
- **搜索空间太大了**：天真的将训练超参数加入到搜索空间中是几乎不可能的，所需要的资源可能非常大。

针对上述几个问题，本文提出了 NARS，主要思想包括三点：

- (1) 为了支持不同约束条件下 NAS 结果的重复利用，本文训练了一个准确率预测器，可能使用预测器在几个 CPU minute 内找到新的约束条件下的架构超参数。
- (2) 为了避免只搜索架构或只搜索超参数的陷阱，预测器同时对架构和训练超参数进行打分。
- (3) 为了避免爆炸式增长的训练时间，我们在代理数据集熵预训练了一个预测器来根据架构表征预测架构的统计信息 (如 FLOPs，参数量)。

在连续执行预测器预训练、约束迭代优化和基于预测器的**进化搜索**之后，NARS 能够生成有良好泛化性的训练超参数和紧凑模型，并且在 ImageNet 上获取了 SOTA。



