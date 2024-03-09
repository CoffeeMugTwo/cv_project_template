"""Module contains the dataloader
"""

import os
from typing import Tuple

import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        img_suffix: str = ".jpg",
        transform = None,
        target_transform = None
    ) -> None:
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self) -> int:
        return len(self.img_labels)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tuple[float]]:
        img_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[idx, 0] + self.img_suffix
        )
        img = read_image(img_path)
        label = self.img_labels.iloc[idx, 1:]
        if self.transform:
            img = self.target_transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
