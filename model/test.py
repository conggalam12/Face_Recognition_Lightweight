import torch
from src import model
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import random as rd

device = "cuda:1"
trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112,96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
def cosin(a,b):
    a = a.squeeze(0)
    b = b.squeeze(0)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    cos = torch.dot(a,b)/(norm_a*norm_b)
    return cos
def load_img(path):
    img = Image.open(path).convert("RGB")
    img = trans(img).unsqueeze(0).to(device)
    return img
if __name__ == "__main__":
    net = model.Shuffle()
    net.load_state_dict(torch.load('/home/ipcteam/congnt/face/face_recognition/CustomModel/model/Custom_Model/1000.ckpt')['net_state_dict'])
    net.to(device)
    net.eval()
    folder_data = "/home/congnt/face/face_recognition/data/data_umd"
    data_path = os.path.join(folder_data,'data_test.csv')
    df = pd.read_csv(data_path)
    count = 0
    count = 0
    with open("/home/congnt/face/face_recognition/CustomModel/Test/test_umd_1.log","a") as file:
        for i in range(len(df)):
            index = rd.randint(0,len(df)-1)
            data = df.iloc[index]
            path_anchor = data['anchor']
            path_pos = data['positive']
            path_neg = data['negative']
            anchor = net(load_img(path_anchor))
            pos = net(load_img(path_pos))
            neg = net(load_img(path_neg))
            cos_pos = cosin(anchor,pos)
            cos_neg = cosin(anchor,neg)
            if cos_pos > cos_neg:
                count+=1
            if i%10000 == 0 and i>0:
                file.write("Test {}/{} : {}%\n".format((i+1),len(df),count*100/(i+1)))
                print("Test {}/{} : {}%".format((i+1),len(df),count*100/(i+1)))
        file.write("Total :{}%".format(count*100/len(df)))
        print("Total :{}%".format(count*100/len(df)))
