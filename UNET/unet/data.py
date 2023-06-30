import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.n_samples = len(image_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.image_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 # (512, 512, 3)
        image = np.transpose(image, (2, 0, 1)) # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image) # to tensor

        """ Reading mask """
        mask = cv2.imread(self.mask_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   # (512, 512)
        mask = np.expand_dims(mask, axis=0)     # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask) # to tensor

        return image, mask
    
    def __len__(self):
        return self.n_samples

