# Vision Transformer-MNIST
Simplified Scratch Pytorch implementation of Vision Transformer (ViT) with detailed steps (Refer to model.py) for learning internal operations. <br> <br>

<ul>
  <li>Scaled-down version of the original ViT architecture from <a href="https://arxiv.org/pdf/2010.11929.pdf">An Image is Worth 16X16 Words</a>. </lr>
   <li> Has only 400k parameters (Original ViT-Base has 86 million). </li>
  <li>Useful for classifying small datasets like MNIST, FashionMNIST, or SVHN. </li>
</ul>  

The model achieves around **99.4%** test Accuracy on MNIST and **93.0%** on FashionMNIST.


Run commands: <br>
python main.py --dset mnist <br>
python main.py --dset fmnist

<br><br>
Transformer Config:

 | <!-- -->    | <!-- -->    |
--- | --- | 
Input Size | 28 |
Patch Size | 4 | 
Sequence Length | 7*7 = 49 |
Embedding Size | 96 | 
Num of Layers | 6 | 
Num of Heads | 4 | 
Forward Multiplier | 2 | 
