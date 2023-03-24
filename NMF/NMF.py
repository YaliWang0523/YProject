import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NMF(nn.Module):
    itr = 0

    def __init__(self, n_user, n_item, k=10, c_vector=1.0, writer=None):
        super(NMF, self).__init__()
        # P contains the user's latent factors
        # Q contains the item's latent factors
        self.writer = writer
        self.k = k
        self.n_user = n_user
        self.n_item = n_item

        self.P = torch.nn.Parameter(torch.rand(n_user, k))
        self.Q = torch.nn.Parameter(torch.rand(n_item, k))

    def __call__(self, train_x):
        user_id = train_x[:, 0]
        item_id = train_x[:, 1]

        q = self.Q.repeat(1, 1)
        indices_item = torch.stack(list(item_id), dim=0)
        vector_Q = torch.index_select(q, 0, indices_item.squeeze())

        p = self.P.repeat(1, 1)
        indices_user = torch.stack(list(user_id), dim=0)
        vector_P = torch.index_select(p, 0, indices_user.squeeze())

        ui_interaction = torch.sum(vector_P * vector_Q, dim=1)
        return ui_interaction

    def loss(self, prediction, target):
        total = F.mse_loss(prediction, target.squeeze())
        return total


def l2_regularize(array):
    loss = torch.sum(array ** 2.0)
    return loss
