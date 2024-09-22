import os
import torch
import argparse
import datetime
from solver import Solver


def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.model_path,  exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    solver = Solver(args)
    solver.train()               # Training function
    solver.plot_graphs()         # Training plots
    solver.test(train=True)      # Testing function


# Print arguments
def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


# Update arguments
def update_args(args):
    args.model_path  = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    args.n_patches   = (args.image_size // args.patch_size) ** 2
    args.is_cuda     = torch.cuda.is_available()  # Check GPU availability

    if args.is_cuda:
        print("Using GPU")
    else:
        print("Cuda not available.")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple and easy to understand PyTorch implementation of Vision Transformer (ViT) from scratch')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store training graphs and tsne plots')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument("--image_size", type=int, default=28, help='image size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument("--n_channels", type=int, default=1, help='number of channels')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset')

    # ViT Arguments
    parser.add_argument("--embed_dim", type=int, default=64, help='dimensionality of the latent space')
    parser.add_argument("--n_attention_heads", type=int, default=4, help='number of heads to use in Multi-head attention')
    parser.add_argument("--forward_mul", type=int, default=2, help='forward multiplier')
    parser.add_argument("--n_layers", type=int, default=6, help='number of encoder layers')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout value')
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
