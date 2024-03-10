# Vision Transformer-MNIST
Simplified Scratch Pytorch implementation of Vision Transformer (ViT) with detailed steps (Refer to model.py) for understanding internal operations. <br> <br>

<ul>
  <li>Scaled-down version of the original ViT architecture from <a href="https://arxiv.org/pdf/2010.11929.pdf">An Image is Worth 16X16 Words</a> for small datasets. </lr>
   <li>Has only 400k parameters (Original ViT-Base has 86 million). </li>
  <li>Supported datasets: MNIST, FashionMNIST, SVHN, and CIFAR10</li>
</ul>  

<br><br>

Run commands: <br>
<table>
  <tr>
    <th>Dataset</th>
    <th>Command</th>
    <th>Test Acc</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>python -u main.py --dataset mnist --epochs 100</td>
    <td>99.4</td>
  </tr>
  <tr>
    <td>Fashion MNIST</td>
    <td>python -u main.py --dataset fmnist</td>
    <td>93.0</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>python -u main.py --dataset svhn --n_channels 3 --image_size 32</td>
    <td>92.0</td>
  </tr>
</table>


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
