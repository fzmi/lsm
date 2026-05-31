# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from tifffile import tifffile
import torch
import numpy as np


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split / "data"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # todo: changed: 1 channel read
        image = np.array(tifffile.imread(self.samples[index]), dtype=np.float32)

        # fixme: 1/8 channel
        # image = torch.from_numpy(image).permute(2, 0, 1) / 65535.0

        # convert into ndarray of [C, H, W]
        image = torch.from_numpy(image).unsqueeze(0) / 65535.0

        # for each dimension, normalise the values between 0 and 1
        # for i in range(image.shape[0]):
        #     image[i, :, :] = (image[i, :, :] - torch.min(image[i, :, :])) / (
        #         torch.max(image[i, :, :]) - torch.min(image[i, :, :])
        #     )

        if self.transform:
            return self.transform(image)
        return image

    def __len__(self):
        return len(self.samples)
