# Vision Transformer for MNIST and CIFAR10
### Simplified Scratch Pytorch implementation of Vision Transformer (ViT) with detailed steps (Refer to <a href="model.py">model.py</a>). <br> <br>

<ul>
  <li>Scaled-down version of the original ViT architecture from <a href="https://arxiv.org/pdf/2010.11929.pdf">An Image is Worth 16X16 Words</a>. </li>
  <li>Has only 200k-800k parameters depending upon the embedding dimension (Original ViT-Base has 86 million). </li>
  <li>Works with small datasets by using a smaller patch size of 4.</li>
  <li>Supported datasets: MNIST, FashionMNIST, SVHN, and CIFAR10.</li>
</ul>  

<br><br>

## Run commands (also available in <a href="scripts.sh">scripts.sh</a>): <br>

<table>
  <tr>
    <th>Dataset</th>
    <th>Run command</th>
    <th>Test Acc</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>python main.py --dataset mnist --epochs 100</td>
    <td><strong>99.5</strong></td>
  </tr>
  <tr>
    <td>Fashion MNIST</td>
    <td>python main.py --dataset fmnist</td>
    <td><strong>92.3</strong></td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>python main.py --dataset svhn --n_channels 3 --image_size 32 --embed_dim 128 </td>
    <td><strong>96.2</strong></td>
  </tr>
  <tr>
    <td>CIFAR10</td>
    <td>python main.py --dataset cifar10 --n_channels 3 --image_size 32 --embed_dim 128 </td>
    <td>82.5 (<strong>86.3</strong> with RandAug)</td>
  </tr>
</table>


<br><br>
## Transformer Config:

<table>
  <tr>
    <th>Config</th>
    <th>MNIST and FMNIST</th>
    <th>SVHN and CIFAR10</th>
  </tr>
  <tr>
    <td>Input Size</td>
    <td> 1 X 28 X 28   </td>
    <td> 3 X 32 X 32  </td>
  </tr>

  <tr>
    <td>Patch Size</td>
    <td>4</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Sequence Length</td>
    <td>7*7 = 49</td>
    <td>8*8 = 64</td>
  </tr>
  <tr>
    <td>Embedding Size </td>
    <td>64</td>
    <td>128</td>
  </tr>
  <tr>
    <td>Parameters </td>
    <td>210k</td>
    <td>820k</td>
  </tr>
  <tr>
    <td>Num of Layers </td>
    <td>6</td>
    <td>6</td>
  </tr>
  <tr>
    <td>Num of Heads </td>
    <td>4</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Forward Multiplier </td>
    <td>2</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Dropout </td>
    <td>0.1</td>
    <td>0.1</td>
  </tr>
</table>

<!--
<br><br>
## Training Graphs:

<table>
  <tr>
    <th>Dataset</th>
    <th>Accuracy</th>
    <th>Loss</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td> <img src="outputs/mnist/graph_accuracy.png"  alt="MNIST_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/mnist/graph_loss.png"  alt="MNIST_loss" width = 500px height = 250px ></td>
  </tr>
  <tr>
    <td>FMNIST</td>
    <td> <img src="outputs/fmnist/graph_accuracy.png"  alt="FMNIST_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/fmnist/graph_loss.png"  alt="FMNIST_loss" width = 500px height = 250px ></td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td> <img src="outputs/svhn/graph_accuracy.png"  alt="SVHN_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/svhn/graph_loss.png"  alt="SVHN_loss" width = 500px height = 250px ></td>
  </tr>
  <tr>
    <td>CIFAR10</td>
    <td> <img src="outputs/cifar10/graph_accuracy.png"  alt="CIFAR10_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/cifar10/graph_loss.png"  alt="CIFAR10_loss" width = 500px height = 250px ></td>
  </tr>
</table>
-->
