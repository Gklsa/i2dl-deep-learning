"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.dataset)

        if self.shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)

        for start_idx in range(0, n, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch = {}
            for idx in batch_indices:
                sample = self.dataset[idx]
                for key, value in sample.items():
                    if key not in batch:
                        batch[key] = []
                    batch[key].append(value)

            for key in batch:
                batch[key] = np.array(batch[key])

            yield batch

    def __len__(self):
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))

        return length