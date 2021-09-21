### EfficientNet

**Paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. CVPR, 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper)**

#### Abstract

本文系统地研究了模型缩放并确定仔细平衡网络深度、宽度和分辨率可以带来更好的性能。基于这一观察，我们提出了一种新的缩放方法，该方法使用简单而高效的复合系数统一缩放深度/宽度/分辨率的所有维度。我们证明了这种方法在扩大 MobileNets 和 ResNet 方面的有效性。

为了更进一步，我们使用神经架构搜索来设计一个新的 baseline 网络并进行缩放以获得一系列模型，称为 EfficientNets，它比以前的 ConvNets 实现了更好的准确性和效率。

#### 1. Introduction

