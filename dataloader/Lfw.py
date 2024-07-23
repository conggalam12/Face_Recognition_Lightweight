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
class LFW_test(Dataset):
    def __init__(self,root):
        super(LFW_test,self).__init__()
        self.imgl = []
        self.imgr = []
        self.label = []
        path_label = os.path.join(root,'test.txt')
        with open(path_label,'r') as file:
            for line in file:
                if len(line.split()) == 3:
                    img_path,id1,id2 = line.split()
                    imgl_path = img_path+'/'+img_path+'_'+'0'*(4-len(id1))+id1+'.jpg'
                    imgr_path = img_path+'/'+img_path+'_'+'0'*(4-len(id2))+id2+'.jpg'
                    label = 1
                else :
                    imgr_path,id1,imgl_path,id2 = line.split()
                    imgl_path = imgr_path+'/'+imgr_path+'_'+'0'*(4-len(id1))+id1+'.jpg'
                    imgr_path = imgl_path+'/'+imgl_path+'_'+'0'*(4-len(id2))+id2+'.jpg'
                    label = -1
                if os.path.exists(root+'/img/'+imgl_path) and os.path.exists(root+'/img/'+imgr_path):
                    self.imgl.append(root+'/img/'+imgl_path)
                    self.imgr.append(root+'/img/'+imgr_path)
                    self.label.append(label)
        self.class_nums = len(np.unique(self.label))
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112,96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
if __name__ == '__main__':
    path_data = '/home/ipcteam/congnt/face/face_recognition/data/data_Lfw'
    test = LFW_test(path_data)
    a = test.__getitem__(1)

