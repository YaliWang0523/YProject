import torch
from sklearn.utils import shuffle
import numpy as np


class Loader():
    current = 0

    def __init__(self, x, y, n_user, n_item, batchsize=256, do_shuffle=True):
        self.x = x
        self.y = y
        self.n_user = n_user
        self.n_item = n_item
        self.batchsize = batchsize
        self.do_shuffle = do_shuffle
        self.batches = range(0, len(self.x), batchsize)
        if self.do_shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        if self.do_shuffle:
            self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.current = 0
        return self

    def __len__(self):
        return int(len(self.x)/self.batchsize)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __next__(self):
        n = self.batchsize
        if self.current + n > len(self.x):
            raise StopIteration
        i = self.current

        user_idx = self.x[i:i + n, 0]
        item_idx = self.x[i:i + n, 1]
        buy_count = self.x[i:i + n, 2]

        train_matrix = np.zeros((self.n_user, self.n_item))
        train_mask_matrix = np.zeros((self.n_user, self.n_item))

        train_matrix[user_idx, item_idx] = buy_count
        train_mask_matrix[user_idx, item_idx] = 1

        # xs = torch.from_numpy(self.x[i:i + n])
        # ys = torch.from_numpy(self.y[i:i + n])
        self.current += n
        return train_matrix, train_mask_matrix
