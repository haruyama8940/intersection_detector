from asyncore import write
from traceback import print_tb
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import load


# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network CNN 3 + FC 2>
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.lstm = nn.LSTM()
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
    # <Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    # <CNN layer>
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            # self.maxpool,
            self.flatten
        )
    # <FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,

        )

    # <forward layer>
    def forward(self, x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2


class deep_learning:
    def __init__(self, n_channel=3, n_action=4):
        # <tensor device choice>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        # self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()
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
        self.intersection_labels = []
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.first_flag = True
        torch.backends.cudnn.benchmark = False
        self.writer = SummaryWriter(log_dir='/home/haru/nav_ws/src/intersection_detector/data/tensorboard/runs')

    def act_and_trains(self, img, intersection_label):

        # <training mode>
        self.net.train()
        self.train_accuracy =0
        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor(
                [intersection_label], dtype=torch.float32, device=self.device)
            
            self.writer.add_graph(self.net,self.x_cat)
            self.writer.close()
            
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
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)

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
        self.train_accuracy +=  torch.sum(torch.max(y_train,1)[1] == torch.max(t_label_train,1)[1]).item()
        print("label :" ,self.train_accuracy , "/",BATCH_SIZE , (self.train_accuracy/len(t_label_train))*100 , "%")
        # <test>
        self.net.eval()
        intersection_training = self.net(x)
        self.writer.add_scalar("loss",loss,self.count)
        self.count += 1
        # print("action=" ,intersection_training[0][0].item() ,"loss=" ,loss.item())
        # print("action=" ,intersection_training.item() ,"loss=" ,loss.item())

        if self.first_flag:
            self.writer.add_graph(self.net,(x))
            self.writer.close()
            self.writer.flush()
        # <reset dataset>
        if self.x_cat.size()[0] > MAX_DATA:
            self.x_cat = torch.empty(1, 3, 48, 64).to(self.device)
            self.t_cat = torch.empty(1, 4).to(self.device)
            self.first_flag = True
            print("reset dataset")

        # return intersection_training.item(), loss.item()
        return torch.max(intersection_training,1)[1].item() , loss.item()

    def act(self, img):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        # <test phase>
        intersection_test = self.net(x_test_ten)

        # print("act = ", intersection_test)
        return torch.max(intersection_test,1)[1]

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


if __name__ == '__main__':
    dl = deep_learning()
