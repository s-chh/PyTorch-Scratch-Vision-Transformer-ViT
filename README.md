# Vision Transformer-MNIST
Unofficial Pytorch implementation of Vision Transformer (ViT) with detailed steps.

The original architecture from [An Image is Worth 16X16 Words](https://arxiv.org/pdf/2010.11929.pdf) has been scaled down for classifying MNIST dataset.

The model achieves around 98.5% test Accuracy on MNIST and can be fine-tuned for further performance gains.

Model HyperParametes:

Input Size | 28 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Channels | 1 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Patch Size | 7 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Sequence Length | 16 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
