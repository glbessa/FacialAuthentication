import os
import glob
import random

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CelebretiesDataset(Dataset):
  def __init__(self, images_path, names, transform=None):
    super().__init__()

    self.images_path = images_path
    self.names = names
    self.names2image_filenames = {}
    self.length = 0
    for name in names:
      self.names2image_filenames[name] = glob.glob(os.path.join(images_path, name, "*.*"))
      self.length += len(self.names2image_filenames[name])
    self.transform = transform

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    random_name = random.choice(self.names)
    random_index = random.randint(0, len(self.names2image_filenames[random_name]) - 1)

    image1 = read_image(self.names2image_filenames[random_name][random_index])
    image2 = None
    label = None

    if idx % 2 == 0:
      # Different class / person
      random_name = random.choice(self.names)
      random_index = random.randint(0, len(self.names2image_filenames[random_name]) - 1)

      image2 = read_image(self.names2image_filenames[random_name][random_index])
      label = torch.tensor([0], dtype=torch.float)
    else:
      # Same class / person
      temp = self.names2image_filenames[random_name].copy()
      temp.pop(random_index)

      image2_filename = random.choice(temp)

      image2 = read_image(image2_filename)
      label = torch.tensor([1], dtype=torch.float)

    if self.transform:
      image1 = self.transform(image1)
      image2 = self.transform(image2)

    return image1, image2, label