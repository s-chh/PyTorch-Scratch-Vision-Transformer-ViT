import torch
from torchvision import datasets
from torchvision import transforms
import os


def get_loader(args):
    tr = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])
    if args.dset == 'mnist':
        train = datasets.MNIST(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr)
        test = datasets.MNIST(os.path.join(args.data_path, args.dset), train=False, download=True, transform=tr)

    elif args.dset == 'fmnist':
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr)
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dset), train=False, download=True, transform=tr)

    else:
        train = datasets.ImageFolder(root=os.path.join(args.data_path, args.dset, 'trainset'), transform=tr)
        test = datasets.ImageFolder(root=os.path.join(args.data_path, args.dset, 'testset'), transform=tr)

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
