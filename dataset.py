import torch
from torch.utils.data import Dataset

class TourDataset(Dataset):
    def __init__(self, tours, num_cities):
        self.num_cities = num_cities
        self.bos_token = num_cities

        tours_tensor = torch.tensor(tours, dtype=torch.long)

        if tours_tensor.ndim != 2:
            assert False # must be a 2d array

        if tours_tensor.shape[1] != num_cities:
            assert False # tours must be complete

        expected = torch.arange(num_cities).expand_as(tours_tensor)
        if not torch.equal(torch.sort(tours_tensor, dim=1).values, expected):
            assert False # must be a permutation

        if not torch.all(tours_tensor[:, 0] == 0):
            assert False # must start at 0

        bos = torch.full(
            (len(tours_tensor), 1),
            self.bos_token,
            dtype=torch.long
        )

        self.inputs = torch.cat(
            [bos, tours_tensor[:, :-1]],
            dim=1
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index] + self.targets[index]