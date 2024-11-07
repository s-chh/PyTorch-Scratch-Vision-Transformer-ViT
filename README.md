# Vision Transformer from Scratch in PyTorch
### Simplified Scratch Pytorch Implementation of Vision Transformer (ViT) with Detailed Steps (Refer to <a href="model.py">model.py</a>)
Results on small datasets like MNIST, CIFAR10, etc., using a smaller patch size. This network is a scaled-down version of the original ViT. <br><br> 


Key Points:
<ul>
  <li>The default network is a scaled-down version of the original ViT architecture from <a href="https://arxiv.org/pdf/2010.11929.pdf">An Image is Worth 16X16 Words</a>. </li>
  <li>Has only 200k-800k parameters depending upon the embedding dimension (Original ViT-Base has 86 million). </li>
  <li>Tested on MNIST, FashionMNIST, SVHN, and CIFAR10 datasets. </li>
  <li>Uses a smaller patch size of 4.</li>
  <li>Can be used with bigger datasets by increasing the model parameters and patch size.</li>
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
    <td><strong>86.3</strong> (82.5 w/o RandAug)</td>
  </tr>
  <tr>
    <td>CIFAR100</td>
    <td>python main.py --dataset cifar100 --n_channels 3 --image_size 32 --embed_dim 128 </td>
    <td><strong>59.6</strong> (55.8 w/o RandAug)</td>
  </tr>
</table>


<br><br>
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
<br>

