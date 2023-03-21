import torch
import numpy as np


class AutoRec(torch.nn.Module):
    def __init__(self, n_user, n_item, k=10, lambda_value=1, writer=None):
        super(AutoRec, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.k = k
        self.lambda_value = lambda_value
        self.writer = writer
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_item, self.k),
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.k, self.n_item)
        )

    def forward(self, train_x):
        encoder = self.encoder(train_x)
        decoder = self.decoder(encoder)
        return decoder

    def loss(self, decoder, input, optimizer, mask_input):
        cost = 0
        temp2 = 0
        cost += ((decoder-input) * mask_input).pow(2).sum()
        loss = cost.clone().detach()

        for i in optimizer.param_groups:
            for j in i['params']:
                if j.data.dim() == 2:
                    temp2 += (j.data).pow(2).sum()
        cost += temp2 * self.lambda_value * 0.5
        rmse = torch.sqrt(loss /
                          (torch.nonzero(mask_input)).size(dim=0))
        return cost, rmse
