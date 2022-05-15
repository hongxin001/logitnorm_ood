from algorithms.base_framework import SingleModel
import torch.nn.functional as F


class Standard(SingleModel):
    def train_batch(self, index, inputs, targets, epoch):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.net(inputs)

        loss = self.loss_function(logits, targets)
        return loss

