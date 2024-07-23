import os
import shutil
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn
from torch.nn import DataParallel
import torch.nn.functional as f
from torchvision import transforms
from datetime import datetime
from config import BATCH_SIZE,TEST_FREQ,SAVE_FREQ, RESUME, SAVE_DIR, TOTAL_EPOCH,LOSS , OPTIMIZE , MODEL,THRESHOLD , DIVIE_LR
from config import CASIA_DATA_DIR,VN_DATA_DIR,DATA
from src import model
from src import model_shuffle
from src import model_mobile
from src import model_resnet
from src import model_suk
from src import loss
from dataloader.Casia_Webface import Casia
from dataloader.Lfw import LFW_test
from dataloader.VN import VN
from dataloader.Test import test
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import time
import numpy as np
import pdb

def cosine_similarity(tensor1, tensor2):

    dot_product = torch.sum(tensor1 * tensor2, dim=1)
    norm1 = torch.norm(tensor1, dim=1)
    norm2 = torch.norm(tensor2, dim=1)

    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

def accuracy(predict,label):
    argmax = torch.argmax(predict,dim = 1)
    accuracy = torch.sum(argmax == label).item()
    return accuracy
if __name__ == '__main__':
    device = 'cuda:1'
    # other init
    print('Run device',device[-1])
    start_epoch = 1

    save_dir = os.path.join(SAVE_DIR, MODEL,DATA,datetime.now().strftime('%Y_%m_%d'))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    save_metric = os.path.join(SAVE_DIR,MODEL,'Metric',DATA,datetime.now().strftime('%Y_%m_%d'))
    if os.path.exists(save_metric):
        shutil.rmtree(save_metric)
    os.mkdir(save_metric)

    path_log = '/home/ipcteam/congnt/face/face_recognition/model/train_log/{}/{}/train_{}.log'.format(MODEL,DATA,datetime.now().strftime('%Y_%m_%d'))
    model_data = None
    if MODEL == 'Custom':
        model_data = "Custom"
    if os.path.exists(path_log):
        os.remove(path_log)
        
    if DATA == 'Casia':
        print('Defining casia dataloader')
        dataset = Casia(root=CASIA_DATA_DIR,model = model_data)
        num_classes = len(os.listdir(CASIA_DATA_DIR+'/img'))
    if DATA == 'Vn':
        print('Defining vn dataloader')
        dataset = VN(root=VN_DATA_DIR,model=model_data)
        num_classes = len(os.listdir(VN_DATA_DIR+'/img'))

    data_test = test(root = VN_DATA_DIR,model=model_data)
    total_far = data_test.far()
    train_ratio = 1
    # val_ratio = 0.1
    num_samples = len(dataset)
    num_train_samples = int(train_ratio * num_samples)
    # num_val_samples = int(val_ratio * num_samples)

    indices = torch.randperm(num_samples)

    train_dataset = Subset(dataset, indices[:num_train_samples])
    # val_dataset = Subset(dataset, indices[num_train_samples:])
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
    #                                           shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test,batch_size=64,
                                              shuffle=True, num_workers=8)
    # define model  
    if MODEL == 'Resnet':
        print('Defind ResnetFace model')
        net = model_resnet.resnet_face18()
    elif MODEL == 'Shuffle':
        print('Defind ShuffleFaceNet model')
        net = model_shuffle.ShuffleFaceNet()
    elif MODEL == 'Mobile':   
        print('Defind MobileFaceNet model')
        net = model_mobile.MobileFaceNet()
    elif MODEL == 'Suk':
        net = model_suk.Suk()
        print('Defind Suk model')
    else:
        print('Defind custom model')
        net = model.Shuffle()

    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = loss.FocalLoss().to(device)
    print("Using {} loss".format(LOSS))
    if LOSS == 'ArcFace':
        metric_fc = loss.ArcMarginProduct(in_features=128,out_features=num_classes,device = device).to(device)
    if LOSS == 'CosFace':
        metric_fc = loss.CosFace(num_classes=num_classes,device=device)
    # optimzer
    print("Using {} optimize".format(OPTIMIZE))
    if OPTIMIZE == 'SGD':
        optimizer= optim.SGD(net.parameters(), lr=0.01,momentum=0.9,weight_decay=5e-4)
    if OPTIMIZE == 'Adam':
        optimizer= optim.Adam(net.parameters(), lr=0.1)
    if RESUME:
        ckpt = torch.load(RESUME)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        path_metric = os.path.join(save_metric, '%03d.ckpt' % ckpt['epoch'])
        ckpt_metric = torch.load(path_metric)
        metric_fc.load_state_dict(ckpt_metric['metric_state_dict'])
    net = net.to(device)

    scheduler = StepLR(optimizer,step_size=20, gamma=0.5)

    optimzer4center = optim.Adam(metric_fc.parameters(), lr=0.01)
    sheduler_4center = StepLR(optimizer,step_size=20, gamma=0.5)
    best_epoch = 0
    for epoch in range(start_epoch, TOTAL_EPOCH+1):       
        # train model 
        
        print('Train Epoch: {}/{} '.format(epoch, TOTAL_EPOCH),end = '\t')
        net.train()
        train_total_loss = 0.0
        total_acc = 0
        total = 0
        since = time.time()
        for iter,data in enumerate(train_loader):
            img, label = data[0].to(device), data[1].to(device).long()
            batch_size = img.size(0)
            # optimizer_ft.zero_grad() 

            raw_logits = net(img)
            if LOSS == 'CosFace':
                logit,mlogits = metric_fc(raw_logits, label)
            else:
                mlogits = metric_fc(raw_logits, label)
            
            total_loss = criterion(mlogits, label)

            optimizer.zero_grad() 
            optimzer4center.zero_grad()
            total_loss.backward()
            
            optimizer.step()
            optimzer4center.step()
            train_total_loss += total_loss.item() * batch_size
            total += batch_size

            acc = accuracy(mlogits.data,label)
            total_acc+=acc
            
        total_acc = total_acc*100/len(train_dataset)
        if epoch in DIVIE_LR:
            scheduler.step()
            sheduler_4center.step()
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        train_total_loss = train_total_loss / total
        loss_msg = 'Train_loss: {:.4f}\t'.format(train_total_loss)
        print(loss_msg,end = '')
        # test val dataset
        # val_total_loss = 0.0
        # total = 0
        # total_acc_val = 0
        # net.eval()
        # for data in val_loader:
        #     img, label = data[0].to(device), data[1].to(device).long()
        #     batch_size = img.size(0)

        #     raw_logits = net(img)

        #     if LOSS == 'CosFace':
        #         logit,mlogits = metric_fc(raw_logits, label)
        #     else:
        #         mlogits = metric_fc(raw_logits, label)
        #     total_loss = criterion(mlogits, label)
            
        #     val_total_loss += total_loss.item() * batch_size
        #     total += batch_size
        #     acc_val = accuracy(mlogits.data,label)
        #     total_acc_val += acc_val
        # val_total_loss = val_total_loss / total
        # total_acc_val = total_acc_val*100/len(val_dataset)
        # loss_msg_val = 'Val_loss: {:.4f}\t'.format(val_total_loss)
        # print(loss_msg_val,end = '')
        net.eval()
        acc_far = 0
        if epoch % TEST_FREQ == 0:
            acc = 0
            for data in test_loader:
                imgl,imgr,label = data[0].to(device),data[1].to(device),data[2].to(device)
                imgl = net(imgl)
                imgr = net(imgr)
                # calculator accuracy
                cosin = cosine_similarity(imgl,imgr)
                mask = cosin<THRESHOLD
                cosin[mask] = -1
                cosin[~mask] = 1
                result = torch.sum(cosin == label)
                 
                #calculator far
                comparison = (cosin == label) & (label == -1)
                far = torch.sum(comparison)
                acc_far += far
                acc+=result
            acc = acc*100/len(data_test)
            acc_far = acc_far/total_far
            acc_far = (1-acc_far)*100

            print('Acc Test: {:.2f}%    Acc Far: {:.2f}%'.format(acc,acc_far))
        else:
            # print('Acc:{:.2f}%    Acc_val:{:.2f}%\n'.format(total_acc,total_acc_val))
            print('Acc:{:.2f}%'.format(total_acc),end = '   ')
        lr = optimizer.param_groups[0]['lr']
        with open(path_log,'a') as file:
            if epoch % TEST_FREQ == 0:
                test = 'Acc Test: {:.2f}%      Acc Far: {:.2f}%'.format(acc,acc_far)
            else:
                # test = 'Acc:{:.2f}%    Acc_val:{:.2f}%'.format(total_acc,total_acc_val)
                test = 'Acc:{:.2f}% '.format(total_acc)
            # file.write('Train Epoch: {}/{} '.format(epoch, TOTAL_EPOCH)+loss_msg + loss_msg_val+test+"    lr : {}\n".format(lr))
            file.write('Train Epoch: {}/{} '.format(epoch, TOTAL_EPOCH)+loss_msg +test+"    lr : {}\n".format(lr))
        end = time.time() - since
        
        print('Time : {:.4f}m  lr {:.4f}\n'.format(end/60,lr))
        if epoch % SAVE_FREQ == 0:
            net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
            metric = metric_fc.state_dict()
            torch.save({
                'epoch': epoch,
                'metric_state_dict': metric},
                os.path.join(save_metric, '%03d.ckpt' % epoch))
    print('finishing training')