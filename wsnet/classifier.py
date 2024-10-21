# coding=gbk
import os
import numpy as np
import torch
from torch import save, no_grad
from tqdm import tqdm
import shutil


class Classifier():
    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    @staticmethod
    def save_checkpoint(state, is_best, checkpoint):
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

        with no_grad():
            for data, target in tqdm(self.test_loader):
                target = torch.squeeze(target)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                # print(pred,target)
                top1 += pred.eq(target.view_as(pred)).sum().item()

        top1_acc = 100. * top1 / len(self.test_loader.sampler)

        return top1_acc

    def train_step(self, criterion, optimizer):
        losses = []
        top1=0
        for data, target in tqdm(self.train_loader, total=len(self.train_loader)):
            target = torch.squeeze(target)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            #计算准确率
            pred=output.argmax(dim=1,keepdim=True)
            top1 += pred.eq(target.view_as(pred)).sum().item()


            loss = criterion(output, target)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc=100. *top1/len(self.train_loader.sampler)

        return losses,train_acc

    def train(self, criterion, optimizer, epochs, scheduler, modelpath):

        best_accuracy = 0.
        losses = []
        test_accuracies = []
        np_losses = []
        train_accuracies=[]

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses,train_accuracy = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']

            test_accuracy = self.test(criterion)
            test_accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)

            if scheduler:
                scheduler.step()
            is_best = test_accuracy > best_accuracy
            if is_best:
                best_accuracy = test_accuracy
                filepath = modelpath + '/cnn_checkpoint_{}%.pth'.format(best_accuracy)  # 保存模型到result文件目录下
                torch.save(self.model, filepath)

            print('Train Epoch {0}\t Loss: {1:.6f}  Train Accuracy{2:.3f}  Test Accuracy {2:.3f}  lr: {3:.4f}'.format(epoch,
                                                                                                  epoch_losses.mean(),train_accuracy,
                                                                                                  test_accuracy, lr))
            np_losses.append(epoch_losses.mean())
            print('Best accuracy: {:.3f} '.format(best_accuracy))

        return np_losses,test_accuracies,train_accuracies
