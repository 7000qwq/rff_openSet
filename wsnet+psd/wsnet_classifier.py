import os
import numpy as np
import torch
from torch import save, no_grad
from tqdm import tqdm
import shutil

NumOfFeatures = 9

class Classifier():
    def __init__(self, model, train_loader1=None, test_loader1=None, device=None):
        super().__init__()
        self.model = model
        self.train_loader1 = train_loader1
        self.test_loader1 = test_loader1
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
            # dataiter2 = iter(self.test_loader2)
            for data1, target1 in tqdm(self.test_loader1):
                target1 = torch.squeeze(target1)
                data1, target1 = data1.to(self.device), target1.to(self.device)
                output = self.model(data1)
                test_loss += criterion(output, target1).item()
                pred = output.argmax(dim=1, keepdim=True)
                # print(pred,target)
                top1 += pred.eq(target1.view_as(pred)).sum().item()

        top1_acc = 100. * top1 / len(self.test_loader1.sampler)

        return top1_acc

    def train_step(self, criterion, optimizer):
        losses = []
        # dataiter2 = iter(self.train_loader2)
        for [data1, target1] in tqdm((self.train_loader1), total=len(self.train_loader1)):
            target1 = torch.squeeze(target1)
            print('targetinfo:',target1.shape)
            print(('tartype:',type(target1)))
            data1,target1 = data1.to(self.device),  target1.to(self.device)
            output = self.model(data1)

            loss = criterion(output, target1)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses

    def train(self, criterion, optimizer, epochs, scheduler):

        best_accuracy = 0.
        losses = []
        accuracies = []
        np_losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']
            test_accuracy = self.test(criterion)
            accuracies.append(test_accuracy)
            if scheduler:
                scheduler.step()
            is_best = test_accuracy > best_accuracy
            if is_best:
                best_accuracy = test_accuracy
                filepath = 'results/sib_256fft_{}%.pth'.format(best_accuracy)
                torch.save(self.model, filepath)

            print('Train Epoch {0}\t Loss: {1:.6f}\t Test Accuracy {2:.3f} \t lr: {3:.4f}'.format(epoch,
                                                                                                  epoch_losses.mean(),
                                                                                                  test_accuracy, lr))
            np_losses.append(epoch_losses.mean())
            print('Best accuracy: {:.3f} '.format(best_accuracy))

        return

