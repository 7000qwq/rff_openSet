import os
import numpy as np
import torch
from torch import save, no_grad
from tqdm import tqdm
import shutil
from matplotlib import pyplot as plt


class Classifier():
    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    @staticmethod
    def save_checkpoint(state, is_best, checkpoint):  # 保存模型
        head, tail = os.path.split(checkpoint)
        if not os.path.exists(head):
            os.makedirs(head)

        filename = os.path.join(head, '{0}_checkpoint.pth.tar'.format(tail))
        save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(head, '{0}_best.pth.tar'.format(tail)))
        return

    def test(self, criterion):
        self.model.eval()
        top1 = 0
        test_loss = 0.

        # with no_grad():  # 冻结梯度，不进行反向传播
        for data, target in tqdm(self.test_loader):
            target = torch.squeeze(target)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)

            top1 += pred.eq(target.view_as(pred)).sum().item()
        print("test target:---------------", end="\n")
        print(target)
        print("tese pred:-----------------")
        print(pred.t())
        top1_acc = 100. * top1 / len(self.test_loader.sampler)
        print('len:', len(self.test_loader.sampler))

        return top1_acc

    def train_step(self, criterion, optimizer):
        losses = []
        train_right = 0
        for data, target in tqdm(self.train_loader, total=len(self.train_loader)):
            target = torch.squeeze(target)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            predict = output.argmax(dim=1, keepdim=True)
            # print(predict,target)
            train_right += predict.eq(target.view_as(predict)).sum().item()

            loss = criterion(output, target)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = 100. * train_right / len(self.train_loader.sampler)
        print("train target:---------------", end="\n")
        print(target)
        print("trian pred:-----------------")
        print(predict.t())
        return losses, train_acc

    def test_train(self, criterion, optimizer):
        llosses = []
        train_right = 0
        for data, target in tqdm(self.train_loader, total=len(self.train_loader)):
            target = torch.squeeze(target)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            predict = output.argmax(dim=1, keepdim=True)
            # print(predict,target)
            train_right += predict.eq(target.view_as(predict)).sum().item()

            loss = criterion(output, target)
            llosses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1 = 0
            test_loss = 0.
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            top1 += pred.eq(target.view_as(pred)).sum().item()

        ttrain_acc = 100. * train_right / len(self.train_loader.sampler)
        ttest_acc=100. * top1 / len(self.test_loader.sampler)
        return llosses, ttrain_acc, ttest_acc

    def train(self, criterion, optimizer, epochs, scheduler):

        best_accuracy = 0.
        losses = []
        accuracies = []
        train_accuracies = []
        np_losses = []
        train_acc = []

        for epoch in range(1, epochs + 1):
            self.model.train()

            epoch_losses, train_accuracy = self.train_step(criterion, optimizer)
            test_accuracy = self.test(criterion)
            # epoch_losses, train_accuracy, test_accuracy = self.test_train(criterion, optimizer)

            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']
            accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)
            if scheduler:
                scheduler.step()
            is_best = test_accuracy > best_accuracy
            if is_best:
                best_accuracy = test_accuracy
                filepath = 'CNN_result/cnn_checkpoint_{}%.pth'.format(best_accuracy)  # 保存模型到result文件目录下
                torch.save(self.model, filepath)

            print(
                'Train Epoch {0}\t Loss: {1:.6f}\t Test Accuracy {2:.3f} \t lr: {3:.4f}\t Train Accuracy {4:.3f}'.format(
                    epoch,
                    epoch_losses.mean(),
                    test_accuracy, lr, train_accuracy))
            np_losses.append(epoch_losses.mean())
            print('Best accuracy: {:.3f} '.format(best_accuracy))
        test_acc = np.array(accuracies)
        train_acc = np.array(train_accuracies)
        loss = np.array(np_losses)

        plt.figure(1)
        plt.plot(test_acc, label='test')
        plt.plot(train_acc, label='train')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.figure(2)
        plt.plot(loss)
        plt.xlabel("Epochs")
        plt.ylabel("Total CrossEntropy Loss per Epoch")
        plt.show()

        return
