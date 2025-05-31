import torch as t
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any, Iterator


class IndexedDataset(Dataset):
    def __init__(self, original_dataset: Dataset):
        self.original_dataset = original_dataset

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        data_item = self.original_dataset[idx]
        return data_item, idx
