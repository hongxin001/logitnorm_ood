from tqdm import tqdm
import abc, os
from models.utils import build_model
import torch
import torch.nn.functional as F
import torch.nn as nn
class SingleModel:
    __metaclass__ = abc.ABCMeta
    def __init__(self, args, device, num_classes):
        self.device = device
        self.args = args
        self.num_classes = num_classes
        # Create model
        self.net = build_model(args.model, num_classes, device,args)
        self.iterations = 0

        self.optimizer_model = torch.optim.SGD(self.net.parameters(), args.learning_rate,
                                                momentum=args.momentum, weight_decay=args.decay)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_model, [80,140], gamma=0.1)

        if args.loss == "ce":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif args.loss == "logit_norm":
            from common.loss_function import LogitNormLoss
            self.loss_function = LogitNormLoss(device, self.args.temp)


    def train(self, train_loader, epoch):
        self.net.train()
        loss_avg = 0.0
        for data, target, index in tqdm(train_loader):
            loss = self.train_batch(index, data, target, epoch)
            # backward

            if len(self.args.gpu) > 1:
                self.optimizer_model.module.zero_grad()
                loss.backward()
                self.optimizer_model.module.step()
            else:
                self.optimizer_model.zero_grad()
                loss.backward()
                self.optimizer_model.step()
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        self.scheduler.step()
        return loss_avg


    @abc.abstractmethod
    def train_batch(self, batch_idx, inputs, targets, epoch):
        pass

    def test(self, test_loader):
        self.net.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for dict in test_loader:
                data, target = dict[0].to(self.device), dict[1].to(self.device)

                # forward
                output = self.net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        return loss_avg / len(test_loader), correct / len(test_loader.dataset)
