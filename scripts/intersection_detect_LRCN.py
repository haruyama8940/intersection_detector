from asyncore import write
from traceback import print_tb
from typing_extensions import Self
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from yaml import load


# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 5000


class Net(nn.Module):
    def __init__(self, n_channel, n_out,input_size,hidden_size,batch_first):
        super().__init__()
    # <Network CNN 3 + FC 2>
        v2 = models.mobilenet_v2()
        v2.classifier[1] = nn.Linear(
            in_features=v2.last_channel, out_features=n_out)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,batch_first=batch_first)
        self.output_layer = nn.Linear(hidden_size,n_out)
    # <CNN layer>
        self.v2_layer = v2
    # <LSTM + OUTPUT>
        self.LSTM_OUT_layer = nn.Sequential(
            self.lstm,
            self.output_layer
        )
    # <forward layer>
    def forward(self, x):
        x1 = self.v2_layer(x)
        x2 = self.LSTM_OUT_layer(x1)
        return x2


class deep_learning:
    def __init__(self, n_channel=3, n_action=4):
        # <tensor device choice>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        print("mobilenetv2 !!")
        self.optimizer = optim.Adam(
            self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        # self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()
        self.transform_train = transforms.Compose([transforms.RandomRotation(15),
                                                   transforms.ColorJitter(brightness=0.3,saturation=0.3)])
                                                   
        self.transform_color = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)
        self.n_action = n_action
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.datas = []
        self.buffer_list = torch.zeros(1, 4).to(self.device)
        self.buffer_size = 7
        self.intersection_labels = []
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.first_flag = True
        torch.backends.cudnn.benchmark = False
        # self.writer = SummaryWriter(log_dir='/home/haru/nav_ws/src/intersection_detector/data/tensorboard/runs')

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

            # self.writer.add_graph(self.net,self.x_cat)
            # self.writer.close()

            self.first_flag = False
        # <to tensor img(x),intersection_label(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        # <(t dim [4]) -> [1,4] >
        t = torch.tensor([intersection_label], dtype=torch.float32,
                         device=self.device)
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)

        # <make dataset>
        #print("train x =",self.x_cat.shape,x.device,"train t = " ,self.t_cat.shape,t.device)
        dataset = TensorDataset(self.x_cat, self.t_cat)
        # <dataloader>
        # train_dataset = DataLoader(
        #     dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)

        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=False)
        # <only cpu>
        # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)

        # <split dataset and to device>
        for x_train, t_label_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            t_label_train.to(self.device, non_blocking=True)
            break
        #print("x_train =", x_train.shape,"t_train =",t_label_train )
        x_train = self.transform_color(x_train)
        # <learning>
        self.optimizer.zero_grad()
        # self.net.zero_grad()
        y_train = self.net(x_train)
        #print("y = ",y_train.shape,"t=",t_label_train.shape)
        loss = self.criterion(y_train, t_label_train)
        loss.backward()
        self.optimizer.step()
        # self.writer.add_scalar("loss",loss,self.count)
        self.train_accuracy += torch.sum(torch.max(y_train, 1)
                                         [1] == torch.max(t_label_train, 1)[1]).item()
        print("label :", self.train_accuracy, "/", BATCH_SIZE,
              (self.train_accuracy/len(t_label_train))*100, "%")
        # <test>
        self.net.eval()
        intersection_training = self.net(x)
        intersection_training_out = torch.max(
            intersection_training, 1)[1].item()

        self.buffer_list = torch.cat(
            [self.buffer_list, intersection_training], dim=0)
        bufferdata_train = torch.argmax(torch.sum(self.buffer_list, dim=0))

        # self.writer.add_scalar("loss", loss, self.count)
        self.count += 1
        if self.count >= self.buffer_size:
            #self.buffer_list = self.buffer_list[1:]
            self.buffer_list = torch.zeros(1, 4).to(self.device)
            self.count = 0

        # if self.first_flag:
        #     self.writer.add_graph(self.net, (x))
        #     self.writer.close()
        #     self.writer.flush()
        # <reset dataset>
        if self.x_cat.size()[0] > MAX_DATA:
            self.x_cat = self.x_cat[1:]
            self.t_cat = self.t_cat[1:]
            #self.first_flag = True

        # return intersection_training.item(), loss.item()
        return intersection_training_out, bufferdata_train, loss.item()

    def act(self, img):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        # <test phase>
        intersection_test = self.net(x_test_ten)
        # print(intersection_test)
        self.buffer_list = torch.cat(
            [self.buffer_list, intersection_test], dim=0)
        bufferdata_test = torch.argmax(torch.sum(self.buffer_list, dim=0))
        #print("buffer_list" ,self.buffer_list, "shape", self.buffer_list.shape,"buffer_sum",bufferdata_test)
        self.count += 1
        if self.count >= self.buffer_size:
            #self.buffer_list = self.buffer_list[1:]
            self.buffer_list = torch.zeros(1, 4).to(self.device)
            self.count = 0
        # print("act = ", intersection_test)
        # torch.max(bufferdata_test,1)[1]
        return torch.max(intersection_test, 1)[1], bufferdata_test

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
