# Vision Transformer from Scratch in PyTorch
### This is a simplified Scratch Pytorch Implementation of Vision Transformer (ViT) with detailed Steps (Refer to <a href="model.py">model.py</a>)

## Overview:
<ul>
  <li>The default network is a scaled-down version of the original ViT architecture from the <a href="https://arxiv.org/pdf/2010.11929.pdf">ViT Paper</a>. </li>
  <li>Has only 200k-800k parameters depending upon the embedding dimension (Original ViT-Base has 86 million). </li>
  <li>Tested on MNIST, FashionMNIST, SVHN, CIFAR10, and CIFAR100 datasets. </li>
  <li>Uses a smaller patch size of 4.</li>
  <li>Can be used with bigger datasets by increasing the model parameters and patch size.</li>
  <li>Option to use PyTorch's inbuilt transformer layers or the implemented one to define the ViT's Encoder.</li>
</ul> 

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
    <td><strong>86.3</strong> (82.5 w/o RandAug)</td>
  </tr>
  <tr>
    <td>CIFAR100</td>
    <td>python main.py --dataset cifar100 --n_channels 3 --image_size 32 --embed_dim 128 </td>
    <td><strong>59.6</strong> (55.8 w/o RandAug)</td>
  </tr>
</table>

<strong>use_torch_transformer_layers</strong> argument (in <a href="https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/cf5c88251c1b1f15b46954fa7013bfc86980ddd6/main.py#L61">main.py</a>) switches between PyTorch's inbuilt transformer layers and the implemented one for defining the Vision Transformer's Encoder (code at <a href="https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/026c5bed8d6dc088b24066510dddc01bde0b163d/model.py#L215">model.py</a>).

## Transformer Config:

<table>
  <tr>
    <th>Config</th>
    <th>MNIST and FMNIST</th>
    <th>SVHN and CIFAR</th>
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
Further optimizing the network can provide additional performance gains.

