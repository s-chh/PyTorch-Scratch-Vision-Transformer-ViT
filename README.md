# Vision Transformer-MNIST
Unofficial Pytorch implementation of Vision Transformer (ViT) with detailed steps.

The network is a scaled-down version of the original architecture from [An Image is Worth 16X16 Words](https://arxiv.org/pdf/2010.11929.pdf) for classifying MNIST dataset.

The model achieves around 98.5% test Accuracy on MNIST and can be fine-tuned for further performance gains.

<br><br>
Transformer HyperParameters:

 | <!-- -->    | <!-- -->    |
--- | --- | 
Input Size | 28 |
Patch Size | 7 | 
Sequence Length | 4*4 = 16 |
Embedding Size | 96 | 
Num of Layers | 4 | 
Num of Heads | 8 | 
Forward Multiplier | 2 | 
