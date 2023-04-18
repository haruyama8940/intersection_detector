from typing_extensions import Self
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

# HYPER PARAM
BATCH_SIZE = 16
MAX_DATA = 2000
FRAME_SIZE = 9
EPOCH_NUM = 20

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network mobilenetv2>
        v2 = models.mobilenet_v2(weights='IMAGENET1K_V1')
        v2.classifier[1] = nn.Linear(
            in_features=1280, out_features=1280)
    #<LSTM + OUTPUT>
        self.lstm = nn.LSTM(input_size=1280,
                            hidden_size=512, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(512, n_out)
    # <CNN layer>
        self.v2_layer = v2

    def forward(self, x):
         # x's dimension: [B, T, C, H, W]

        # frameFeatures' dimension: [B, T, CNN's output dimension(1280)]
        frameFeatures = torch.empty(size=(x.size()[0], x.size()[1], 1280), device='cuda')
        for t in range(0, x.size()[1]):
            #<x[B,T,C,H,W]->CNN[B,T,1280]>
            #print("forword_x:",x.shape)
            frame = x[:,t, :, :,:]
            #print("forword_frame:",frame.shape)
            frame_feature = self.v2_layer(frame)
            #print(frame_feature.shape)
            #[B,seq_len,H]
            frameFeatures[:,t, :] = frame_feature
        #<CNN[B,T,1280] -> lstm[B,1280,512]>
        #print("lstm_in:",frameFeatures.shape)
        lstm_out,_ = self.lstm(frameFeatures)
        #<lstm[B,1280]-> FC[B,4,512]>
        # class_out = self.output_layer(lstm_out[:,-1,:])
        print("lstm_out:",lstm_out.shape)
        # for f in range(0,lstm_out.size()[0]):
        class_out = self.output_layer(lstm_out)
        print("class_out",class_out.shape)
        class_out =torch.mean(class_out,dim=1)
        #print("class_out_mean",class_out.shape)
        return class_out


class deep_learning:
    def __init__(self, n_channel=3, n_action=8):
        # <tensor device choice>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        print("mobilenetv2 + LSTM = LRCN_all")
        #print(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        # self.optimizer = optim.SGD()
        #self.optimizer = torch.optim.SGD(self.net.parameters(),lr=0.003,momentum=0.9)
        # self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()
        self.transform_train = transforms.Compose([transforms.RandomRotation(15),
                                                   transforms.ColorJitter(brightness=0.3, saturation=0.3)])

        self.transform_color = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)
        self.n_action = n_action
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.acc_list = []
        self.datas = []
        self.buffer_list = torch.zeros(1, 8).to(self.device)
        self.buffer_size = 7
        self.intersection_labels = []
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.first_flag = True
        self.first_test_flag = True
        self.first_time_flag = True
        torch.backends.cudnn.benchmark = False
        self.writer = SummaryWriter(log_dir='/home/rdclab/orne_ws/src/intersection_detector/runs')
        torch.manual_seed(0)
        torch.autograd.set_detect_anomaly(True)
        self.loss_all = 0.0
        self.intersection_test = torch.zeros(1,8).to(self.device)
        self.old_label = [0,0,0,0,0,0,0,0]
        self.diff_flag = False

    def make_dataset(self, img, intersection_label):
        self.device = torch.device('cpu')
        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor(
                [intersection_label], dtype=torch.float32, device=self.device)
            if self.first_time_flag:
                self.x_cat_time = torch.zeros(1,FRAME_SIZE,3,48,64).to(self.device) 
                self.t_cat_time = torch.clone(self.t_cat)
            # torch.zeros(1,FRAME_SIZE,4).to(self.device)

            self.first_flag = False
            self.first_time_flag = False
    # <to tensor img(x),intersection_label(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        # <(t dim [4]) -> [1,4] >
        t = torch.tensor([intersection_label], dtype=torch.float32,
                         device=self.device)
        if intersection_label == self.old_label:
            self.diff_flag = False
            self.x_cat = torch.cat([self.x_cat, x], dim=0)
            print("cat x_cat!!",self.x_cat.shape)
        else:
            self.first_flag = True
            self.diff_flag = True
            print("change label")
        # <self.x_cat (B,C,H,W) = (8,3,48,64))>
        self.old_label = intersection_label
        # <self.t_cat (B,Size) = (8,4))>
        # self.t_cat = torch.cat([self.t_cat, t], dim=0)
        #print("x_cat:",self.x_ca
        
        if self.x_cat.size()[0] == FRAME_SIZE  and self.diff_flag ==False:
            # <self.x_cat_time (B,T,C,H,W) = (8,8,3,48,64))>
            print("make dataset")
            print("t_data:",t)
            #print("x_cat_time:",self.x_cat_time.shape,"x_cat_sq:",self.x_cat.unsqueeze(0).shape)
            self.x_cat_time = torch.cat((self.x_cat_time, self.x_cat.unsqueeze(0)), dim=0)
            #<self.t_cat_time (B,T,Size) = (8,8,4)>
            self.t_cat_time = torch.cat((self.t_cat_time,t),dim=0)
            # self.x_cat = self.x_cat[1:]
            # self.t_cat = self.t_cat[1:]
            self.first_flag = True
    # <make dataset>
        print("train x =",self.x_cat_time.shape,x.device,"train t = " ,self.t_cat_time.shape,t.device)
        dataset = TensorDataset(self.x_cat_time, self.t_cat_time)
        # train_dataset = DataLoader(
        #     dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=False,num_workers=2)

        return dataset,len(dataset) # ,train_dataset

    def load_dataset(self,dataset):
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=False,num_workers=2)
    
        return train_dataset
    
    def training(self, train_dataset):
        self.device = torch.device('cuda')
        print(self.device)
    # <training mode>
        self.net.train()
        self.train_accuracy = 0
        # dataset , train_dataset =self.make_dataset(img,intersection_label)
        
    # <split dataset and to device>
        for epoch in range(EPOCH_NUM):
            print('epoch', epoch) 
            for x_train,t_label_train in train_dataset:
                # if i == random_train:
                # x_train,t_label_train = train_dataset
                # x_train.to(self.device, non_blocking=True)
                # t_label_train.to(self.device, non_blocking=True)
                x_train = x_train.to(self.device,non_blocking=True)
                t_label_train = t_label_train.to(self.device, non_blocking=True)
        # <use transform>
            # x_train = self.transform_color(x_train)
            #x_train = self.transform_train(x_train)
        # <learning>
                self.optimizer.zero_grad()
                y_train = self.net(x_train)
                loss_all = self.criterion(y_train, t_label_train)
                print("y = ",y_train.shape,"t=",t_label_train)
                self.train_accuracy += torch.sum(torch.max(y_train, 1)
                                                    [1] == torch.max(t_label_train, 1)[1]).item()
                print("epoch:",epoch, "accuracy :", self.train_accuracy, "/",len(t_label_train),
                        (self.train_accuracy/len(t_label_train))*100, "%","loss :",loss_all.item())
                self.writer.add_scalar("loss", loss_all.item(), self.count)
                loss_all.backward()
                self.optimizer.step()
                self.loss_all = loss_all.item()
                self.count += 1
                self.train_accuracy = 0
        print("Finish learning!!")
        finish_flag = True

        return self.train_accuracy, self.loss_all

    def test(self, img):
        self.net.eval()
    # <to tensor img(x)>
        if self.first_test_flag:
            self.x_cat_test = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat_test = self.x_cat_test.permute(0, 3, 1, 2)
            self.first_test_flag = False
        
        x = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        self.x_cat_test = torch.cat([self.x_cat_test, x], dim=0)
        print("x_test_cat:",self.x_cat_test.shape)
        if self.x_cat_test.size()[0] == FRAME_SIZE:
            # self.x_cat_time_test = self.x_cat_test.unsqueeze(0)
            # torch.cat((self.x_cat_time, self.x_cat_test.unsqueeze(0)), dim=0)
            #self.first_test_flag = True
            self.intersection_test = self.net(self.x_cat_test.unsqueeze(0))
            print("s:",self.intersection_test.shape)
            self.x_cat_test = self.x_cat_test[1:]
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
    # <test phase>
            # self.intersection_test = self.net(self.x_cat_test.unsqueeze(0))
        
        return torch.max(self.intersection_test, 1)[1].item()

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')

    def load(self, load_path):
        # <model load>
        self.net.load_state_dict(torch.load(load_path))
        print(load_path)


if __name__ == '__main__':
    dl = deep_learning()
