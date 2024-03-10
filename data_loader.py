import torch
from torchvision import datasets
from torchvision import transforms
import os


def get_loader(args):
    if args.dataset == 'mnist':
        train_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=2, padding_mode='edge'), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    elif args.dataset == 'fmnist':
        train_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=2, padding_mode='edge'), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=2, padding_mode='edge'), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.SVHN(os.path.join(args.data_path, args.dataset), split='train', download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.SVHN(os.path.join(args.data_path, args.dataset), split='test', download=True, transform=test_transform)

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=4, padding_mode='edge'), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train = datasets.CIFAR10(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        test = datasets.CIFAR10(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)

    else:
        print("Unknown dataset")
        exit(0)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size * 2,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)

    return train_loader, test_loader
