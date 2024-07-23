import numpy as np
import cv2 as cv
import os
import imageio
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from sklearn import preprocessing
import torch
import scipy.misc
import shutil
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("..")
class Casia(Dataset):
    def __init__(self, root,model = None):
        super(Casia, self).__init__()
        self.image_list = []
        self.label_list = []
        self.path_label = root+'/'+'label.txt'
        with open(self.path_label,'r') as file:
            for line in file:
                image_path,label = line.split()
                self.image_list.append(root+'/img/'+image_path)
                self.label_list.append(int(label))
        le = preprocessing.LabelEncoder()
        self.label_list = le.fit_transform(self.label_list)
        self.class_nums = len(np.unique(self.label_list))
        if model == "Custom" or model == 'Custom':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112,96)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
        else:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112,112)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
