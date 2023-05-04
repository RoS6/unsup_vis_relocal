import torch
import numpy as np
import pandas as pd
import glob
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image 
from torchvision import datasets, models, transforms
import math 
import random 
import cv2
from PIL import Image
import pickle
import imageio
import cv2
import kornia as K
import kornia.geometry as KG
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from numpy.matlib import repmat


base_dir = 'code/unspervised_vis_loc_reproduce/'

class Carla_Dataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = glob.glob(os.path.join(img_dir, '*.npz'))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.extrinsic_list = []
        self.intrinsic = 0
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels[idx])
        img_path = self.img_labels[idx]
        image_info = np.load(img_path)
        rgb_image = image_info['rgb']
        depth_image = image_info['depth']
        intrinsic = image_info['K']
        extrinsic = image_info['extrinsic']
        self.intrinsic = torch.from_numpy(intrinsic)
        self.extrinsic_list.append(torch.from_numpy(extrinsic))

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        return torch.from_numpy(rgb_image), torch.from_numpy(depth_image)

    
