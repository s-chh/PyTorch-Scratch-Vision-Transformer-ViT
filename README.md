# Vision Transformer from Scratch in PyTorch
### Simplified Scratch Pytorch Implementation of Vision Transformer (ViT) with detailed steps (code at <a href="model.py">model.py</a>)

## Overview:
- The default network is a Scaled-down of the original Vision Transformer (ViT) architecture from the [ViT Paper](https://arxiv.org/pdf/2010.11929.pdf).
- Has only 200k-800k parameters depending upon the embedding dimension (Original ViT-Base has 86 million).
- Tested on Common Datasets: MNIST, FashionMNIST, SVHN, CIFAR10, and CIFAR100.
  - Uses 4×4 patch size for creating longer sequences for small image sizes.
- Can be used with bigger datasets by increasing the model parameters and patch size.
- Option to switch between PyTorch’s inbuilt transformer layers and implemented layers one to define the ViT.

## Usage

Run the following commands to train the model on supported datasets:
```bash
# Train on MNIST
python main.py --dataset mnist --epochs 100

# Train on CIFAR10 with custom embedding size
python main.py --dataset cifar10 --n_channels 3 --image_size 32 --embed_dim 128
```

- View more commands in [`scripts.sh`](scripts.sh).
- Adjust configurations for datasets, image size, and embedding dimensions as needed.

#### Key Argument: [`--use_torch_transformer_layers`](https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/cf5c88251c1b1f15b46954fa7013bfc86980ddd6/main.py#L61")
- Use PyTorch's inbuilt Transformer layers (code [here](https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/026c5bed8d6dc088b24066510dddc01bde0b163d/model.py#L215)):
```bash
python main.py --dataset fmnist --use_torch_transformer_layers
```
- If not using `--use_torch_transformer_layers`, the custom-implemented layers are used instead.

## Datasets and Performance
The model has been tested on multiple datasets with the following results:

| Dataset      | Run Command | Test Accuracy   |
|--------------|-------------|-----------------|
| MNIST        | `python main.py --dataset mnist --epochs 100` | **99.5** |
| FashionMNIST | `python main.py --dataset fmnist` | **92.3** |
| SVHN         | `python main.py --dataset svhn --n_channels 3 --image_size 32 --embed_dim 128` | **96.2** |
| CIFAR10      | `python main.py --dataset cifar10 --n_channels 3 --image_size 32 --embed_dim 128` | **86.3** (82.5 w/o RandAug) |
| CIFAR100     | `python main.py --dataset cifar100 --n_channels 3 --image_size 32 --embed_dim 128` | **59.6** (55.8 w/o RandAug) |

<br>

The following curves show the training and validation accuracy and loss for MNIST. 
| Accuracy Curve | Loss Curve |
| --- | --- |
<img src="outputs/mnist/graph_accuracy.png" width="300"></img> | <img src="outputs/mnist/graph_loss.png" width="300"></img>

For the accuracy and loss curves of all other datasets, refer to the [outputs](outputs/)  folder.

## Model Configurations
Below are the key configurations for the Vision Transformer:

| Parameter             | MNIST / FMNIST  | SVHN / CIFAR    |
|-----------------------|-----------------|-----------------|
| **Input Size**        | 1 × 28 × 28     | 3 × 32 × 32     |
| **Patch Size**        | 4               | 4               |
| **Sequence Length**   | 49              | 64              |
| **Embedding Size**    | 64              | 128             |
| **Parameters**        | 210k            | 820k            |
| **Number of Layers**  | 6               | 6               |
| **Number of Heads**   | 4               | 4               |
| **Forward Multiplier**| 2               | 2               |
| **Dropout**           | 0.1             | 0.1             |

