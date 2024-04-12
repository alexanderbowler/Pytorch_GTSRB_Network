import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset

class TrafficSignsDataset(Dataset):
    def __init__(self, labels_file:str, img_dir:str, transform=None, target_transform=None):
        self.labels = pd.read_csv(labels_file)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.type(torch.float32)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        print(image.shape)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
                