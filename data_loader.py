import torch
from torchvision import datasets, transforms
import os


def get_loader(args):
    if args.dataset == 'mnist':
        # Transforms for train
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        # Transforms for test
        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    elif args.dataset == 'fmnist':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=2), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])
        train = datasets.SVHN(os.path.join(args.data_path, args.dataset), split='train', download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])
        test = datasets.SVHN(os.path.join(args.data_path, args.dataset), split='test', download=True, transform=test_transform)

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),  # RandAugment augmentation for strong regularization
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train = datasets.CIFAR10(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        test = datasets.CIFAR10(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)

    else:
        print("Unknown dataset")
        exit(0)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.n_workers,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.n_workers,
                                                drop_last=False)

    return train_loader, test_loader
