import numpy as np
import cv2 as cv
import os
import imageio
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import torch
import scipy.misc
import shutil

import sys
sys.path.append("..")
class VN(Dataset):
    def __init__(self, root,model = None):
        super(VN, self).__init__()
        self.image_list = []
        self.label_list = []
        self.path_label = root+'/'+'label.txt'
        with open(self.path_label,'r') as file:
            for line in file:
                image_path,label = line.split()
                self.image_list.append(root+'/img/'+image_path)
                self.label_list.append(int(label))

        if model == "Custom" or model == 'Custom':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112,96)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
        else:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112,112)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
    def __getitem__(self, index):
        target = torch.tensor(self.label_list[index],dtype=torch.long)
        img_path = self.image_list[index]
        img = cv.imread(img_path)
        input = self.trans(img)
        return input, target
    def __len__(self):
        return len(self.image_list)
if __name__ == '__main__':
    pass
