import torch
from sklearn.utils import shuffle


class Loader():
    current = 0

    def __init__(self, x, y, batchsize=1024, do_shuffle=True):

        self.shuffle = shuffle
        self.do_shuffle = do_shuffle
        self.x = x
        self.y = y
        self.batchsize = batchsize
        self.batches = range(0, len(self.y), batchsize)
        if do_shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        if self.do_shuffle:
            self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.current = 0
        return self

    def __len__(self):
        # Return the number of batches
        return int(len(self.x) / self.batchsize)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __next__(self):
        n = self.batchsize
        if self.current + n >= len(self.y):
            raise StopIteration
        i = self.current

        xs = torch.from_numpy(self.x[i:i + n])
        ys = torch.from_numpy(self.y[i:i + n])

        self.current += n

        return xs, ys
