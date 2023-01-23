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
BATCH_SIZE = 8
MAX_DATA = 2000
FRAME_SIZE = 8


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network mobilenetv2>
        v2 = models.mobilenet_v2()
        v2.classifier[1] = nn.Linear(
            in_features=v2.last_channel, out_features=v2.last_channel)
    #<LSTM + OUTPUT>
        self.lstm = nn.LSTM(input_size=v2.last_channel,
                            hidden_size=512, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(512, n_out)
    # <CNN layer>
        self.v2_layer = v2

    def forward(self, x):
         # x's dimension: [B, C, T, H, W]

        # frameFeatures' dimension: [B, T, CNN's output dimension(1280)]
        frameFeatures = torch.empty(size=(x.size()[0], x.size()[1], 1280), device='cuda')
        for t in range(0, x.size()[1]):
            #<x[B,C,T,H,W]->CNN[B,T,1280]>
            #print("forword_x:",x.shape)
            frame = x[:,t, :, :,:]
            #print("forword_frame:",frame.shape)
            frame_feature = self.v2_layer(frame)
            #print(frame_feature.shape)
            #[B,seq_len,H]
            frameFeatures[:,t, :] = frame_feature
        #<CNN[B,T,1280] -> lstm[B,1280,512]>
        lstm_out,_ = self.lstm(frameFeatures)
        #<lstm[B,1280]-> FC[B,4,512]>
        # class_out = self.output_layer(lstm_out[:,-1,:])
        #print("lstm_out:",lstm_out.shape)
        # for f in range(0,lstm_out.size()[0]):
        class_out = self.output_layer(lstm_out)
        print("class_out",class_out.shape)
        class_out =torch.mean(class_out,dim=1)
        print("class_out_mean",class_out.shape)
        return class_out


class deep_learning:
    def __init__(self, n_channel=3, n_action=4):
        # <tensor device choice>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        print("mobilenetv2 + LSTM = LRCN_all")
        # print(self.net)
        self.optimizer = optim.Adam(
            self.net.parameters(), eps=1e-2, weight_decay=5e-4)
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
        self.buffer_list = torch.zeros(1, 4).to(self.device)
        self.buffer_size = 7
        self.intersection_labels = []
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.first_flag = True
        torch.backends.cudnn.benchmark = False
        self.writer = SummaryWriter(log_dir='/home/rdclab/catkin_ws/src/intersection_detector/runs')
        torch.manual_seed(0)
        torch.autograd.set_detect_anomaly(True)
        loss = 0.0

    def act_and_trains(self, img, intersection_label):

    # <training mode>
        self.net.train()
        self.train_accuracy = 0
        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor(
                [intersection_label], dtype=torch.float32, device=self.device)
            self.x_cat_time = torch.zeros(1,FRAME_SIZE,3,48,64).to(self.device) 
            self.t_cat_time = torch.clone(self.t_cat)
            # torch.zeros(1,FRAME_SIZE,4).to(self.device)

            self.first_flag = False
    # <to tensor img(x),intersection_label(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        # <(t dim [4]) -> [1,4] >
        t = torch.tensor([intersection_label], dtype=torch.float32,
                         device=self.device)
        
        # <self.x_cat (B,C,H,W) = (8,3,48,64))>
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        # <self.t_cat (B,Size) = (8,4))>
        # self.t_cat = torch.cat([self.t_cat, t], dim=0)
        print("x_cat:",self.x_cat.shape,"t_cat",self.t_cat.shape)
        if self.x_cat.size()[0] == FRAME_SIZE:
            # <self.x_cat_time (B,T,C,H,W) = (8,8,3,48,64))>
            #print("x_cat_time:",self.x_cat_time.shape,"x_cat_sq:",self.x_cat.unsqueeze(0).shape)
            self.x_cat_time = torch.cat((self.x_cat_time, self.x_cat.unsqueeze(0)), dim=0)
            #<self.t_cat_time (B,T,Size) = (8,8,4)>
            self.t_cat_time = torch.cat((self.t_cat_time,t),dim=0)
            self.x_cat = self.x_cat[1:]
            # self.t_cat = self.t_cat[1:]
    # <make dataset>
        print("train x =",self.x_cat_time.shape,x.device,"train t = " ,self.t_cat_time.shape,t.device)
        dataset = TensorDataset(self.x_cat_time, self.t_cat_time)
        #dataset = TensorDataset(self.x_cat_time, self.t_cat_time)
    # <dataloader>
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)

        # train_dataset = DataLoader(
        #     dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=False)
        # #dataset_random = random.randint(0,len(train_dataset)-1)
        # random_train = random.randint(0,(len(train_dataset)-1))
        # print("random:",random_train,"dataset_len:",len(train_dataset))
        # train_iter = train_dataset._iterator
        # for _ in range(0,random_train):
        #     train_iter.next()
    # <split dataset and to device>
        # for i in train_dataset:
        for x_train,t_label_train in train_dataset:
            # if i == random_train:
            # x_train,t_label_train = train_dataset
            x_train.to(self.device, non_blocking=True)
            t_label_train.to(self.device, non_blocking=True)
            break

        #x_train , t_label_train = train_dataset.shape [random.randint(0,len(train_dataset))]
        #print("x_train =", x_train.shape,"t_train =",t_label_train.shape)
    # <use transform>
        # x_train = self.transform_color(x_train)
        # x_train = self.transform_train(x_train)
    # <learning>
        # self.optimizer.zero_grad()
        # self.net.zero_grad
        self.optimizer.zero_grad()
        y_train = self.net(x_train)
        loss_all = self.criterion(y_train, t_label_train)
        print("y = ",y_train.shape,"t=",t_label_train.shape)
        #     # self.writer.add_scalar("loss",loss,self.count)
        #print("y_train:", y_train)
        self.train_accuracy += torch.sum(torch.max(y_train, 1)
                                            [1] == torch.max(t_label_train, 1)[1]).item()
        print("accuracy :", self.train_accuracy, "/",len(t_label_train),
                   (self.train_accuracy/len(t_label_train))*100, "%")
        #     print("accuracy :", self.train_accuracy, "/", BATCH_SIZE*(t+1),
        #         (self.train_accuracy/BATCH_SIZE*(t+1))*100, "%")
        #     # self.train_accuracy += torch.sum(torch.max(y_train[0,:,:], 1)
        #     #                                 [1] == torch.max(t_label_train[0,:,:], 1)[1]).item()
        #     # print("label :", self.train_accuracy, "/", BATCH_SIZE,
        #     #     (self.train_accuracy/len(t_label_train[0,:,:]))*100, "%")
        # loss_all = torch.sum(self.loss_list)
        # #print("loss_all:",loss_all.item())
        # loss_batch = loss_all.item()
        # loss_all.backward(retain_graph=True)
        self.writer.add_scalar("loss", loss_all.item(), self.count)
        self.optimizer.step()
        
    
    # <test>
        self.net.eval()
        intersection_training = self.net(x.unsqueeze(0))
        print("intersection_out:",intersection_training.shape)
        intersection_training_out = torch.max(
            intersection_training, 1)[1].item()

        # self.buffer_list = torch.cat(
        #     [self.buffer_list, intersection_training], dim=0)
        # bufferdata_train = torch.argmax(torch.sum(self.buffer_list, dim=0))
        # bufferdata_train =intersection_training
        # self.writer.add_scalar("loss", loss, self.count)
        self.count += 1

        # if self.first_flag:
        #     self.writer.add_graph(self.net, (x))
        #     self.writer.close()
        #     self.writer.flush()
        # <reset dataset>
        if self.x_cat_time.size()[0] > MAX_DATA:
            self.x_cat_time = self.x_cat_time[1:]
            self.t_cat_time = self.t_cat_time[1:]
            #self.first_flag = True

        # return intersection_training.item(), loss.item()
        # return intersection_training_out, loss_all.item()
        return intersection_training_out, loss_all.item()

    def act(self, img):
        self.net.eval()
    # <to tensor img(x)>
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
    # <test phase>
        intersection_test = self.net(x_test_ten.unsqueeze(0))
        # print(intersection_test)
        # self.buffer_list = torch.cat(
        #     [self.buffer_list, intersection_test], dim=0)
        # bufferdata_test = torch.argmax(torch.sum(self.buffer_list, dim=0))
        #print("buffer_list" ,self.buffer_list, "shape", self.buffer_list.shape,"buffer_sum",bufferdata_test)
        self.count += 1
        if self.count >= self.buffer_size:
            #self.buffer_list = self.buffer_list[1:]
            self.buffer_list = torch.zeros(1, 4).to(self.device)
            self.count = 0
        # print("act = ", intersection_test)
        # torch.max(bufferdata_test,1)[1]
        #return torch.max(intersection_test, 1)[1]
        return torch.max(intersection_test, 1)[1].item()
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
