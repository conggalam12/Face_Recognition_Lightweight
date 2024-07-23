import torch
import torch.nn as nn
import math
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # Reshape the tensor
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # Transpose the tensor
        x = x.transpose(1, 2).contiguous()

        # Reshape back to the original shape
        x = x.view(batch_size, num_channels, height, width)

        return x
class IRShuffleUnit_d(nn.Module):
    def __init__(self, main_channels, side_channels):
        super(IRShuffleUnit_d, self).__init__()
        self.main = main_channels
        self.side = side_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.main[0],self.main[1],kernel_size=1),
            nn.BatchNorm2d(self.main[1]),
            nn.PReLU(),
        )
        self.DWconv_2 = nn.Sequential(
            nn.Conv2d(self.main[1],self.main[2],kernel_size=3,stride=2,groups=self.main[1],padding=1),
            nn.BatchNorm2d(self.main[2]),
            nn.PReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.main[2],self.main[3],kernel_size=1),
            nn.BatchNorm2d(self.main[3]),
        )
        self.DWconv_4 = nn.Sequential(
            nn.Conv2d(self.side[0],self.side[1],kernel_size=3,stride=2,groups=self.side[0],padding=1),
            nn.BatchNorm2d(self.side[1]),
            nn.PReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(self.side[1],self.side[2],kernel_size=1),
            nn.BatchNorm2d(self.side[2]),
        )
        self.channel_shuffle = ChannelShuffle(groups=3)
    def forward(self, x):
        # Main branch
        x_m = x
        x_s = x
        x_add = x_m
        x_m = self.conv_1(x_m)
        x_m = self.DWconv_2(x_m)
        x_m = self.conv_3(x_m)
        # if x_m.size(1) == x_add.size(1):
        #     x_m = x_m+x_add[:,:,::2,::2] + x_add[:,:,1::2,1::2]
        # else:
        #     x_m = x_m+x_add[:,::2,::2,::2] +x_add[:,1::2,::2,::2]+ x_add[:,::2,1::2,1::2] + x_add[:,1::2,1::2,1::2]
        x_s = self.DWconv_4(x_s)
        x_s = self.conv_5(x_s)

        out = torch.cat((x_m,x_s),dim = 1)

        out = self.channel_shuffle(out)
        return out

class IRShuffleUnit_c(nn.Module):
    def __init__(self, main_channels):
        super(IRShuffleUnit_c, self).__init__()
        self.main = main_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.main[0],self.main[1],kernel_size=1),
            nn.BatchNorm2d(self.main[1]),
            nn.PReLU(),
        )
        self.DWconv_2 = nn.Sequential(
            nn.Conv2d(self.main[1],self.main[2],kernel_size=3,groups=self.main[1],padding=1),
            nn.BatchNorm2d(self.main[2]),
            nn.PReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.main[2],self.main[3],kernel_size=1),
            nn.BatchNorm2d(self.main[3]),
        )
        self.channel_shuffle = ChannelShuffle(groups=3)
    def forward(self, x):
        num_channels = x.size(1)
        # split
        x_s = x[:,num_channels//2:,:,:]
        x_m = x[:,:num_channels//2,:,:]
        x_add = x_m
        x_m = self.conv_1(x_m)
        x_m = self.DWconv_2(x_m)
        x_m = self.conv_3(x_m)
        x_m = x_m+x_add
        out = torch.cat((x_m,x_s),dim = 1)

        out = self.channel_shuffle(out)
        return out

class Shuffle(nn.Module):
    def __init__(self):
        super(Shuffle,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.PWConv1 = nn.Conv2d(192,512,kernel_size=1)
        self.DWConv1 = nn.Conv2d(512,512,kernel_size=(7,6),groups=512)
        self.PWConv2 = nn.Conv2d(512,128,kernel_size=1)
        self.ir_shuffle1 = IRShuffleUnit_d([32,96,96,16],[32,64,32])
        self.ir_shuffle2 = IRShuffleUnit_c([24,48,48,24])
        self.ir_shuffle3 = IRShuffleUnit_d([48,192,192,48],[48,96,48]) 
        self.ir_shuffle4 = IRShuffleUnit_c([48,96,96,48])
        self.ir_shuffle5 = IRShuffleUnit_c([48,96,96,48])
        self.ir_shuffle6 = IRShuffleUnit_c([48,96,96,48])
        self.ir_shuffle7 = IRShuffleUnit_d([96,256,256,96],[96,192,96])
        self.ir_shuffle8 = IRShuffleUnit_c([96,192,192,96])
    def forward(self,image):
        out = self.conv1(image)
        out = self.ir_shuffle1(out)
        out = self.ir_shuffle2(out)
        out = self.ir_shuffle3(out)
        out = self.ir_shuffle4(out)
        out = self.ir_shuffle5(out)
        out = self.ir_shuffle6(out)
        out = self.ir_shuffle7(out)
        out = self.ir_shuffle8(out)

        out = self.PWConv1(out)
        out = self.DWConv1(out)
        out = self.PWConv2(out)
        out = out.squeeze(dim=-2).squeeze(dim=-1)
        return out
def convert_onnx():
    net = Shuffle()
    net.eval()
    input = torch.randn(1,3,112,96)
    net.load_state_dict(torch.load('/home/ipcteam/congnt/face/face_recognition/model/weights/Custom/Vn/190.ckpt')['net_state_dict'])
    torch.onnx.export(net,
                      input,
                      'custom.onnx',
                      opset_version=11,    
                      input_names=['input'],
                      output_names=['output'])

def test_data():
    device = 'cuda:1'
    class VN_test(Dataset):
        def __init__(self,root):
            super(VN_test,self).__init__()
            self.imgl = []
            self.imgr = []
            self.label = []
            path_label = os.path.join(root,'test.txt')
            with open(path_label,'r') as file:
                for line in file:
                    imgl_path,imgr_path,label = line.split()
                    self.imgl.append(root+'/img/'+imgl_path)
                    self.imgr.append(root+'/img/'+imgr_path)
                    self.label.append(label)
            self.class_nums = len(np.unique(self.label))
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112,96)),
                # transforms.RandomHorizontalFlip(),
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
    net = Shuffle()
    net.to(device)
    net.eval()
    best = 0
    best_epoch = 0
    threshold = 0.7
    while(threshold>0.3):
        for i in range(2,61):
            model_id = i*5
            if i<20:
                model_id = '0'+str(model_id)
            else:
                model_id = str(model_id)
            mode_path = '/home/ipcteam/congnt/face/face_recognition/model/weights/Custom/Vn/2024_03_05/' +model_id + '.ckpt'
            net.load_state_dict(torch.load(mode_path)['net_state_dict'])
            data_test = VN_test(root = '/home/ipcteam/congnt/face/face_recognition/data/data_VN') 
            test_loader = DataLoader(data_test,batch_size=128,
                                                    shuffle=True, num_workers=8)
            
            acc = 0
            for data in test_loader:
                imgl,imgr,label = data[0].to(device),data[1].to(device),data[2].to(device)
                imgl = net(imgl)
                imgr = net(imgr)
                cosin = F.cosine_similarity(imgl,imgr)
                mask = cosin<=threshold
                cosin[mask] = -1
                cosin[~mask] = 1
                result = torch.sum(cosin == label)
                acc+=result
            acc = acc*100/len(data_test)
            if best<acc:
                best = acc
                best_epoch = model_id
            print('Acc : {:.2f}%        Epoch : {}\n'.format(acc,model_id))
        print("Best epoch : {} ----- acc : {}".format(best_epoch,best))
        threshold -=0.01
        with open('/home/ipcteam/congnt/face/face_recognition/report.txt','a') as file:
            file.write("Best epoch : {} ----- acc : {} ----threshold :{}\n".format(best_epoch,best,threshold))
def test_cos():
    trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112,96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    def cosine_similarity(tensor1, tensor2):

        dot_product = torch.sum(tensor1 * tensor2, dim=1)
        norm1 = torch.norm(tensor1, dim=1)
        norm2 = torch.norm(tensor2, dim=1)

        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity
    def load_img(img_path):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = trans(img).unsqueeze(0)
        return img
    
    device = 'cuda:1'
    
    anchor = load_img('/home/ipcteam/congnt/face/face_recognition/data/data_VN/img/385/53.png').to(device)
    pos = load_img('/home/ipcteam/congnt/face/face_recognition/data/data_VN/img/385/18.png').to(device)
    neg = load_img('/home/ipcteam/congnt/face/face_recognition/data/data_VN/img/807/8.png').to(device)
    net = Shuffle()
    net.to(device)
    net.load_state_dict(torch.load('/home/ipcteam/congnt/face/face_recognition/model/weights/Custom/Vn/2024_03_12/025.ckpt')['net_state_dict'])
    net.eval()
    with torch.no_grad():
        output_anchor = net(anchor)
        output_pos = net(pos)
        output_neg= net(neg)
        cosin_pos = cosine_similarity(output_anchor,output_pos)
        cosin_neg = cosine_similarity(output_anchor,output_neg)
        print("Positive cos:",cosin_pos.item())
        print("Negative cos:",cosin_neg.item())
if __name__ == "__main__": 
    test_cos()
    # net = Shuffle()
    # net.eval()
    # net.load_state_dict(torch.load('/home/ipcteam/congnt/face/face_recognition/model/weights/Custom/Vn/500.ckpt')['net_state_dict'])
    # torch.save(net.state_dict(),'model.pt')
    #test_data()
    # convert_onnx()
        

    
    
