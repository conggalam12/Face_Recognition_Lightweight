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

class test(Dataset):
    def __init__(self,root,model = None):
        super(test,self).__init__()
        self.imgl = []
        self.imgr = []
        self.label = []
        path_label = os.path.join(root,'test.txt')
        with open(path_label,'r') as file:
            for line in file:
                imgl_path,imgr_path,label = line.split()
                self.imgl.append(root+'/img/'+imgl_path)
                self.imgr.append(root+'/img/'+imgr_path)
                self.label.append(int(label))
        self.class_nums = len(np.unique(self.label))
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
    def load_img(self,path):
        img = cv.imread(path)
        img = self.trans(img)
        return img
    def __getitem__(self, index):
        target = np.int32(self.label[index]) # -1 if neg and 1 is pos
        imgl_path = self.imgl[index]
        imgr_path = self.imgr[index]
        imgl = self.load_img(imgl_path)
        imgr = self.load_img(imgr_path)
        return imgl,imgr,target
    def __len__(self):
        return len(self.imgl)
    def far(self):
        tensor_label = torch.tensor(self.label)
        return torch.sum(tensor_label == -1)
