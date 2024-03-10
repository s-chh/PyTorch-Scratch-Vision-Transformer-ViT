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

<table>
  <tr>
    <th>Config</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Input Size</td>
    <td>28</td>
  </tr>
  <tr>
    <td>Patch Size</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Sequence Length</td>
    <td>7*7 = 49</td>
  </tr>
  <tr>
    <td>Embedding Size </td>
    <td>96</td>
  </tr>
  <tr>
    <td>Num of Layers </td>
    <td>6</td>
  </tr>
  <tr>
    <td>Num of Heads </td>
    <td>4</td>
  </tr>
  <tr>
    <td>Forward Multiplier </td>
    <td>2</td>
  </tr>
  <tr>
    <td>Epochs </td>
    <td>100</td>
  </tr>

</table>

