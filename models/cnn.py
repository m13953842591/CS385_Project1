try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
from config import *
import os
from data_utils.data_loader import FaceClassificationDataset, categorical_accuracy
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import numpy as np
import glob


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def get_middle_output(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x.cpu()


class CNN:
    def __init__(self):
        self.net = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = ToTensor()
        self.name = "cnn"

    def get_middle_output(self, x):
        return self.net.get_middle_output(x)

    def load_last_checkpoint(self, checkpoint_dir):

        cp_list = glob.glob(
            os.path.join(checkpoint_dir, r"cnn_epoch*.pth"))
        if not cp_list:
            print("No checkpoint found, using default cnn")
            self.net = AlexNet(num_classes=2)
            return 0
        max_e = 0
        max_cp = None
        for cp in cp_list:
            e = int(cp.split('_')[-3])
            if e > max_e:
                max_e = e
                max_cp = cp
        print("max_cp = ", max_cp)
        self.net = torch.load(max_cp, map_location=self.device)
        self.net.eval()
        print("successfully retored from checkpoint %s" % max_cp)
        return max_e

    def train(self, epoch=10, batch_size=128, lr=0.001, checkpoint_dir=None, auto_resume=True):
        latest_epoch = 0
        if auto_resume and checkpoint_dir is not None:
            latest_epoch = self.load_last_checkpoint(checkpoint_dir)
        else:
            self.net = AlexNet(2)
            self.net.to(self.device)
            print("fresh start!")

        print("loading training and testing data...")
        train_dataset = FaceClassificationDataset(
            root_dir=os.path.join(DATA_PATH, 'classification/train'))
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)
        test_dataset = FaceClassificationDataset(
            root_dir=os.path.join(DATA_PATH, 'classification/test'))
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0)
        print("loading finished")

        opt = optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        train_losses = [0.0] * epoch
        train_acces = [0.0] * epoch
        test_losses = [0.0] * epoch
        test_acces = [0.0] * epoch

        train_batch = len(train_loader)
        test_batch = len(test_loader)

        for e in range(epoch):

            training_loss, training_acc = 0.0, 0.0
            for batch_ndx, data in enumerate(train_loader):
                # get the inputs

                inputs, labels = data['image'].to(self.device), data['label'].to(
                    self.device)
                # zero the parameter gradients
                opt.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()
                # print statistics

                training_loss += loss.item()
                training_acc += categorical_accuracy(outputs, labels).item()
                if batch_ndx % 20 == 0:
                    print("epoch[%d], batch[%d], loss=%.4f, acc=%.4f" % (
                          e+latest_epoch+1, batch_ndx, loss.item(),
                          categorical_accuracy(outputs, labels).item()))

            testing_loss, testing_acc = 0.0, 0.0
            for batch_ndx, data in enumerate(test_loader):
                # get the inputs
                inputs, labels = data['image'].to(self.device), data['label'].to(
                    self.device)
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                testing_loss += loss.item()
                testing_acc += categorical_accuracy(outputs, labels).item()

            train_losses[e] = training_loss / train_batch
            train_acces[e] = training_acc / train_batch
            test_losses[e] = testing_loss / test_batch
            test_acces[e] = testing_acc / test_batch

            print('<summary> epoch = [%d], train_loss = %.3f, train_acc = %.3f,'
                  ' test_loss = %.3f, test_acc = %.3f '
                  % (e+latest_epoch+1, train_losses[e], train_acces[e],
                     test_losses[e], test_acces[e]))

            print("saving networks...")
            save_path = os.path.join(WORKSPACE, "checkpoints",
                                     "cnn_epoch_%02d_acc_%.4f.pth" % (
                                         e+latest_epoch+1, test_acces[e]))
            torch.save(self.net.state_dict(), save_path)

        print('Finished Training')

        plt.subplot(2, 1, 1)
        plt.plot(np.arange(latest_epoch+1, latest_epoch+1+epoch), train_losses,
                 color='blue', label='training loss')
        plt.plot(np.arange(latest_epoch+1, latest_epoch+1+epoch), test_losses,
                 color='red', label='testing loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(latest_epoch+1, latest_epoch+1+epoch), train_acces,
                 color='blue', label='training acc')
        plt.plot(np.arange(latest_epoch+1, latest_epoch+1+epoch), test_acces,
                 color='red', label='testing acc')
        plt.legend()

        plt.xlabel('rate')
        plt.ylabel('epochs')
        plt.savefig(
            'cnn_epoch%d_to_%d.png' % (latest_epoch + 1, latest_epoch + epoch))

    def predict_proba(self, x_sample):
        # The input x_sample is a single image in shape(H, W, C)
        input = self.transform(x_sample)
        c, h, w = input.size()
        output = self.net(input.view(-1, c, h, w).to(self.device))
        output = functional.softmax(output, dim=1)
        return output[0][1].item()

    def predict(self, x, threshold = 0.99):
        # The input x_sample is a single image in shape(H, W, C)
        if len(x.shape) == 3:
            # single sample
            pred = self.predict_proba(x) > threshold
            return pred
        else:
            preds = []
            for x_sample in x:
                pred = self.predict_proba(x_sample) > threshold
                preds.append(pred)
            return np.array(preds)

    def score(self):
        test_dataset = FaceClassificationDataset(
            root_dir=os.path.join(DATA_PATH, 'classification/test'))
        test_loader = DataLoader(test_dataset,
                                 batch_size=1000,
                                 shuffle=True,
                                 num_workers=0)
        batch = next(iter(test_loader))
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # get the inputs
        inputs, labels = batch['image'].to(self.device), batch['label'].to(
            self.device)
        outputs = self.net(inputs)
        loss = criterion(outputs, labels)
        testing_acc = categorical_accuracy(outputs, labels).item()
        return testing_acc


if __name__ == '__main__':
    cnn = CNN()
    cnn.train()
    # cnn.load_last_checkpoint(os.path.join(WORKSPACE, "checkpoints"))
    print("acc = ", cnn.score())
