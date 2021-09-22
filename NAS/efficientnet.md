### EfficientNet

**Paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. CVPR, 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper)**

#### Abstract

本文系统地研究了模型缩放并确定仔细平衡网络深度、宽度和分辨率可以带来更好的性能。基于这一观察，我们提出了一种新的缩放方法，该方法使用简单而高效的复合系数统一缩放深度/宽度/分辨率的所有维度。我们证明了这种方法在扩大 MobileNets 和 ResNet 方面的有效性。

为了更进一步，我们使用神经架构搜索来设计一个新的 baseline 网络并进行缩放以获得一系列模型，称为 EfficientNets，它比以前的 ConvNets 实现了更好的准确性和效率。

#### 1. Introduction

缩放 ConvNets 被广泛用来实现更高的准确率，但是这个过程并没有被很好的理解，目前有很多相关的方法，常见的方式是缩放网络的深度或宽度，以及通过图像分辨率来缩放模型，在之前的工作中，大都是缩放这 3 个维度中的一个，虽然任意缩放 3 个维度中的 2 个或 3 个是可能的，但这不仅需要大量手工调整，而且实现的准确率和效率也不是最优的。

本文作者研究和重新思考了缩放 ConvNets 的过程，其中的中心问题是：是否有一种原则性的方法来缩放 ConvNets 以实现更好的精度和效率？本文的研究表明：平衡网络的深度/宽度/分辨率非常关键，并且这种平衡可以通过用一个常数比例对他们进行放大实现。

本文的复合缩放方法使用一组固定的系数统一缩放网络的深度、宽度和分辨率，例如，如果我们想使用 $2^N$ 倍的计算资源，可以简单对网络进行如下缩放：深度放大 ${\alpha}^N$，宽度放大 ${\beta}^N$，图像分辨率方法 ${\gamma}^N$，其中 $\alpha, \beta, \gamma$ 是通过在原始小型网络上进行小网格搜索确定的，下图说明了本文的复合缩放方法与常见方法的区别。

![image-20210922095706279](../../note/_image/image-20210922095706279.png)

#### 2. Related Work

从卷积神经网络的精度、效率、模型缩放三个方面介绍了一些相关工作。

#### 3. Compound Model Scaling

##### 3.1. Problem Formulation

作者将一个 ConvNet 定义为：
$$
\mathcal{N}=\bigodot_{i=1 \ldots s} \mathcal{F}_{i}^{L_{i}}\left(X_{\left\langle H_{i}, W_{i}, C_{i}\right\rangle}\right)
$$
其中将网络分为 *s* 个 stage，每个 stage 有 $L_i$ 层，每层有相同的基本模块 $F_i$ (除了下采样层有些不同)，$\mathcal{F}_{i}^{L_{i}}$ 表示在第 *i* 个 stage 模块 $F_i$ 重复了 $L_i$ 次，其输入为 $X_{\left\langle  H_{i}, W_{i}, C_{i}\right\rangle}$ 

对于一个 baseline 网络，需要对其进行缩放，模块 $F_i$ 已经固定，需要改变的是模型的深度 $(L_i)$，宽度 $(C_i)$，分辨率 $(H_i, W_i)$，为了降低搜索空间，作者**约束所有层的同一维度以相同比例进行缩放**，最终可以看作一个优化问题：
$$
\begin{array}{ll}
\max _{d, w, r} & \operatorname{Accuracy}(\mathcal{N}(d, w, r)) \\
\text { s.t. } & \mathcal{N}(d, w, r)=\bigodot_{i=1 \ldots s} \hat{\mathcal{F}}_{i}^{d \cdot \hat{L}_{i}}\left(X_{\left\langle r \cdot \hat{H}_{i}, r \cdot \hat{W}_{i}, w \cdot \hat{C}_{i}\right\rangle}\right) \\
& \operatorname{Memory}(\mathcal{N}) \leq \text { target\_memory } \\
& \operatorname{FLOPS}(\mathcal{N}) \leq \text { target\_flops }
\end{array}
$$
其中 $d, w, r$ 分别是网络深度、宽度和分辨率的缩放系数，$\hat{\mathcal{F}}_{i}, \hat{L}_{i}, \hat{H}_{i}, \hat{W}_{i}, \hat{C}_{i}$ 是在 baseline 网络中预定义好的参数，本文的 baseline 网络就是后文的 EfficientNet-B0.

##### 3.2. Scaling Dimensions

对于上述问题，作者先在单独维度上进行缩放，如下图所示，**单一维度的缩放在精度达到 80% 后就迅速饱和**。

![image-20210922155902630](../../note/_image/image-20210922155902630.png)

##### 3.3. Compound Scaling

作者观察到，不同维度的缩放并不是独立的，直观地，对于更高分辨率的图像，我们应该增加网络深度，使得更大的感受野可以帮助获取在更大图像中相似的特征，相应地，我们也应该增加网络的宽度，使得可以获取更多细粒度的特征。这样的观察**需要协调和平衡不同的缩放维度**。

![image-20210922170847945](../../note/_image/image-20210922170847945.png)

上图中的实验结果表明：**相同 FLOPS 成本下，同时缩放多个维度比单独缩放一个维度能够实现效果更好**。

本文提出了一种新的**复合缩放方法**，使用一个复合系数 $\phi$ 来统一缩放网络深度、宽度和分辨率：
$$
\begin{aligned}
\text{depth: } &d=\alpha^{\phi} \\
\text{width: } &w=\beta^{\phi} \\
\text{resolution: } &r=\gamma^{\phi} \\
\text { s.t. } &\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 \\
\quad &\alpha \geq 1, \beta \geq 1, \gamma \geq 1
\end{aligned}
$$
其中 $\alpha, \beta, \gamma$ 是通过进行小网格搜索确定的常数，$\phi$ 就是用户指定的一个系数，来根据可用资源控制模型缩放到什么程度。

注意：常规卷积操作的 FLOPS 与 $d, w^2, r^2$ 线性相关，即网络深度扩大至 2 倍，FLOPS也增加至 2 倍；网络宽度和分辨率扩大至 2 倍，FLOPS 则增加至 4 倍，由于 ConvNets 的计算量主要在于卷积操作，使用公式 (3) 对网络进行缩放 FLOPS 会增加至 $(\alpha \cdot \beta^{2} \cdot \gamma^{2})^{\phi}$ 倍，本文作者限制 $\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$ ，使得总的 FLOPS 大约增加至 $2^{\phi}$ 倍。

#### 4. EfficientNet Architecture

由于模型缩放不会改变操作类型/基本模块，因此具有良好的 baseline 网络至关重要。我们使用现有的 ConvNets 将评估本文的缩放方法，但为了更好地证明本文缩放方法的有效性，我们还发开了一个新的 mobile-size 的 baseline，称为 EfficientNet.

受到 MnasNet 的启发，我们**利用同时优化准确率和 FLOPS 多目标神经架构搜索来开发 baseline 网络**。具体来说，我们使用和 MnasNet 相同的搜索空间，使用 $ACC(m) \times {[FLOPS(m)/T]}^w$ 作为优化目标 (因为不针对硬件设备，所以优化 FLOPS 而不是延迟)，其中 $T$ 是目标 FLOPS，$w=-0.07$ 是控制准确率和 FLOPS 平衡的超参数。我们通过搜索得到了一个高效的网络，将其命名为 EfficientNet-B0，具体结果如下图。

![image-20210922165320445](../../note/_image/image-20210922165320445.png)

从 EfficientNet-B0 开始，我们应用复合缩放方法获取一系列模型：

- **STEP 1**：固定 $\phi=1$，根据公式 (2) 和 (3) 对 $\alpha, \beta, \gamma$ 进行小网格搜索，最终，我们发现在 $\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$ 的限制下， EfficientNet-B0 的最佳缩放系数是：$\alpha=1.2, \beta=1.1, \gamma=1.15$.
- **STEP 2**：固定 $\alpha, \beta, \gamma$，根据公式 (3) 使用不同的 $\phi$ 对 EfficientNet-B0 进行缩放，得到 一系列网络，命名为 EfficientNet-B1 到 B7 (论文中没有给出这些网络使用的 $\phi$ 是多少).

#### 5. Experiments

- 用现有的 ConvNets 测试本文提出的复合缩放方法，说明了复合缩放方法相比于单一维度的缩放效果更好。
- 在 ImageNet 和迁移学习数据集上对 EfficientNet-B0 到 B7 进行评估，结果就是准确率高、模型小、速度快。

#### Conclusion

本文提出了一种针对 ConvNets 的复合缩放方法 (主要创新点)，并且使用 MnasNet 类似的 NAS 方法搜索到了一个基线网络 EfficientNet-B0，然后应用本文的复合缩放方法得到了 EfficientNet-B1 到 B7，最后的实验说明了这些网络的优越性。
