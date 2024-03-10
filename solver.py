import torch
import torch.nn as nn
import os
from torch import optim
from model import VisionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader

class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)

        self.model = VisionTransformer(n_channels=self.args.n_channels, embed_dim=self.args.embed_dim, 
                                        n_layers=self.args.n_layers, n_attention_heads=self.args.n_attention_heads, 
                                        forward_mul=self.args.forward_mul, image_size=self.args.image_size, 
                                        patch_size=self.args.patch_size, n_classes=self.args.n_classes)
        
        if self.args.is_cuda:
            print("Using GPU")
            self.model = self.model.cuda()
        else:
            print("Cuda not available.")

        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'ViT_model.pt')))

        self.ce = nn.CrossEntropyLoss()

    def test_dataset(self, loader):
        self.model.eval()

        actual = []
        pred = []

        for (x, y) in loader:
            if self.args.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                logits = self.model(x)
            predicted = torch.max(logits, 1)[1]

            actual += y.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred)
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm

    def test(self, train=True):
        if train:
            acc, cm = self.test_dataset(self.train_loader)
            print(f"Train acc: {acc:.2%}\nTrain Confusion Matrix:")
            print(cm)

        acc, cm = self.test_dataset(self.test_loader)
        print(f"Test acc: {acc:.2%}\nTest Confusion Matrix:")
        print(cm)

        return acc

    def train(self):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.args.warmup_epochs, end_factor=1.0, total_iters=self.args.warmup_epochs, last_epoch=-1, verbose=True)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs-self.args.warmup_epochs, eta_min=1e-5, verbose=True)

        best_acc = 0
        for epoch in range(self.args.epochs):

            self.model.train()

            for i, (x, y) in enumerate(self.train_loader):
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                logits = self.model(x)
                loss = self.ce(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print(f'Ep: {epoch+1}/{self.args.epochs}, It: {i+1}/{iter_per_epoch}, loss: {loss:.4f}')

            test_acc = self.test(train=((epoch+1)%25==0)) # Test training set every 25 epochs
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

            torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "ViT_model.pt"))
            
            if epoch < self.args.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()
