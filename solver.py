import torch
import torch.nn as nn
import os
from torch import optim
from model import Transformer
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader

class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)

        self.model = Transformer(args).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []

        if db.lower() == 'train':
            loader = self.train_loader
        elif db.lower() == 'test':
            loader = self.test_loader

        for (imgs, labels) in loader:
            imgs = imgs.cuda()

            with torch.no_grad():
                class_out = self.model(imgs)
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm

    def test(self):
        train_acc, cm = self.test_dataset('train')
        print("Tr Acc: %.2f" % (train_acc))
        print(cm)

        test_acc, cm = self.test_dataset('test')
        print("Te Acc: %.2f" % (test_acc))
        print(cm)
    
        return train_acc, test_acc

    def train(self):
        total_iters = 0
        best_acc = 0
        iter_per_epoch = len(self.train_loader)
        test_epoch = max(self.args.epochs // 10, 1)

        optimizer = optim.Adam(self.model.parameters(), self.args.lr, weight_decay=1e-5)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        
        for epoch in range(self.args.epochs):

            self.model.train()

            for i, (imgs, labels) in enumerate(self.train_loader):
                total_iters += 1

                imgs, labels = imgs.cuda(), labels.cuda()

                logits = self.model(imgs)
                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, total_iters: %d, err: %.4f'
                          % (epoch + 1, self.args.epochs, i + 1, iter_per_epoch, total_iters, clf_loss))

            if (epoch + 1) % test_epoch == 0:
                test_acc, cm = self.test_dataset('test')
                print("Test acc: %0.2f" % (test_acc))
                print(cm)

                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(self.model.state_dict(), os.path.join(self.args.model_path, 'Transformer.pt'))

            cos_decay.step()