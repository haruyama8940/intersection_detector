from typing_extensions import Self
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset,Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

#test
from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import binary_accuracy
from torcheval.metrics import BinaryAccuracy
# HYPER PARAM
BATCH_SIZE = 32
MAX_DATA = 2000
# FRAME_SIZE = 10
FRAME_SIZE = 16
EPOCH_NUM = 30


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network CNN 3 + FC 2>
    #     v2 = models.mobilenet_v2()
    #     v2.classifier[1] = nn.Linear(
    #         in_features=v2.last_channel, out_features=n_out)
    # # <CNN layer>
    #     self.v2_layer = v2
        v3 = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        v3.classifier[-1]= nn.Linear(in_features=1280, out_features = n_out)
    # <CNN layer>
        self.v3_layer = v3
    # <forward layer>
    def forward(self, x):
        class_out = self.v2_layer(x)
   
        return class_out
    
class bag_to_tensor:
    def __init__(self):
        # <tensor device choice>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.net = Net(n_channel=3, n_out=8)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        self.totensor = transforms.ToTensor()
        self.transform_train = transforms.Compose([transforms.RandomRotation(10),
                                                   transforms.ColorJitter(brightness=0.3, saturation=0.3)])
        self.normalization = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_color = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.acc_list = []
        self.datas = []
        self.buffer_list = torch.zeros(1, 8).to(self.device)
        self.buffer_size = 7
        self.intersection_labels = []
        # # balance_weights = torch.tensor([1.0, 1.0,5.0,5.0,1.0,5.0,10.0,5.0]).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight=balance_weights)
        # balance_weights = torch.tensor([1.0, 1.0,5.0,5.0,1.0,5.0,10.0,5.0]).to(self.device)
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
        # self.device = torch.device('cpu')
        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor(
                [intersection_label], dtype=torch.float32, device=self.device)
            self.first_flag = False
    # <to tensor img(x),intersection_label(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        # <(t dim [4]) -> [1,4] >
        t = torch.tensor([intersection_label], dtype=torch.float32,
                         device=self.device)
         # print('\033[32m'+'test_mode'+'\033[0m')
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)
    # <make dataset>
        print("train x =",self.x_cat.shape,x.device,"train t = " ,self.t_cat.shape,t.device)

        return self.x_cat,self.t_cat

    def cat_tensor(self, image_1_path, image_2_path, image_3_path, label_1_path, label_2_path,label_3_path):
        load_1_image_tensor = torch.load(image_1_path)
        load_2_image_tensor = torch.load(image_2_path)
        load_3_image_tensor = torch.load(image_3_path)

        load_1_label_tensor = torch.load(label_1_path)
        load_2_label_tensor = torch.load(label_2_path)
        load_3_label_tensor = torch.load(label_3_path)

        cat_image_tensor = torch.cat((load_1_image_tensor,load_2_image_tensor,load_3_image_tensor),dim=0)
        cat_label_tensor = torch.cat((load_1_label_tensor,load_2_label_tensor,load_3_label_tensor),dim=0)
        print(cat_image_tensor.shape)
        return cat_image_tensor,cat_label_tensor
    
    def cat_training(self, load_x_tensor,load_t_tensor):
        # self.device = torch.device('cuda')
        print(self.device)
        # load_x_tensor = torch.load(image_path)
        # load_t_tensor = torch.load(label_path)
        # print("x_tensor:",load_x_tensor,"t_tensor:",load_t_tensor)
        print("label info :",torch.sum(load_t_tensor ,dim=0))
        dataset = TensorDataset(load_x_tensor, load_t_tensor)
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)
    # <training mode>
        self.net.train()
        self.train_accuracy = 0
    
    # <split dataset and to device>
        for epoch in range(EPOCH_NUM):
            print('epoch', epoch) 
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batch_loss = 0.0
            batch_accuracy = 0.0
            for x_train,t_label_train in train_dataset:
                # if i == random_train:
                # x_train,t_label_train = train_dataset
                # x_train.to(self.device, non_blocking=True)
                # t_label_train.to(self.device, non_blocking=True)
                x_train = x_train.to(self.device,non_blocking=True)
                t_label_train = t_label_train.to(self.device, non_blocking=True)
        # <use transform>
                # print("ddd=",x_train[0,:,:,:,:].shape)
            # for i in range(BATCH_SIZE):
                # x_train = self.transform_color(x_train)
                # self.transform_train(x_train[i,:,:,:,:])
                # x_train =self.normalization(x_train)
        # <learning>
                self.optimizer.zero_grad()
                y_train = self.net(x_train)
                loss_all = self.criterion(y_train, t_label_train)
                # print("y = ",y_train.shape,"t=",t_label_train)
                self.train_accuracy += torch.sum(torch.max(y_train, 1)
                                                    [1] == torch.max(t_label_train, 1)[1]).item()
                print("epoch:",epoch, "accuracy :", self.train_accuracy, "/",len(t_label_train),
                        (self.train_accuracy/len(t_label_train))*100, "%","loss :",loss_all.item())
                self.writer.add_scalar("loss", loss_all.item(), self.count)
                self.writer.add_scalar("accuracy",(self.train_accuracy/len(t_label_train))*100,self.count)
                loss_all.backward()
                self.optimizer.step()
                self.loss_all = loss_all.item()
                batch_loss += self.loss_all
                batch_accuracy += multiclass_accuracy(
                    input=torch.max(y_train,1)[1],
                    target=torch.max(t_label_train,1)[1],
                    num_classes=8,
                    average="micro"
                ).item()

                self.count += 1
                self.train_accuracy = 0
            epoch_loss = batch_loss / len(train_dataset)
            epoch_accuracy = batch_accuracy /len(train_dataset)
            print("epoch loss:",epoch_loss,"epoch accuracy:",epoch_accuracy)
            self.writer.add_scalar("epoch loss", epoch_loss, epoch)
            self.writer.add_scalar("epoch accuracy",epoch_accuracy,epoch)
        print("Finish learning!!")
        finish_flag = True

        return self.train_accuracy, self.loss_all
    def training(self, image_path,label_path):
        # self.device = torch.device('cuda')
        print(self.device)
        load_x_tensor = torch.load(image_path)
        load_t_tensor = torch.load(label_path)
        # print("x_tensor:",load_x_tensor,"t_tensor:",load_t_tensor)
        print("label info :",torch.sum(load_t_tensor ,dim=0))
        dataset = TensorDataset(load_x_tensor, load_t_tensor)
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=False)
    # <training mode>
        self.net.train()
        self.train_accuracy = 0
    
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
                # print("ddd=",x_train[0,:,:,:,:].shape)
            # for i in range(BATCH_SIZE):
                # x_train = self.transform_color(x_train)
                # self.transform_train(x_train[i,:,:,:,:])
                # x_train =self.normalization(x_train)
        # <learning>
                self.optimizer.zero_grad()
                y_train = self.net(x_train)
                loss_all = self.criterion(y_train, t_label_train)
                # print("y = ",y_train.shape,"t=",t_label_train)
                self.train_accuracy += torch.sum(torch.max(y_train, 1)
                                                    [1] == torch.max(t_label_train, 1)[1]).item()
                print("epoch:",epoch, "accuracy :", self.train_accuracy, "/",len(t_label_train),
                        (self.train_accuracy/len(t_label_train))*100, "%","loss :",loss_all.item())
                self.writer.add_scalar("loss", loss_all.item(), self.count)
                self.writer.add_scalar("accuracy",(self.train_accuracy/len(t_label_train))*100,self.count)
                loss_all.backward()
                self.optimizer.step()
                self.loss_all = loss_all.item()
                self.count += 1
                self.train_accuracy = 0
        print("Finish learning!!")
        finish_flag = True

        return self.train_accuracy, self.loss_all

    def model_test(self, image_path,label_path):
        self.net.eval()
        accuracy =0.0
        test_accuracy = 0.0
    # <to tensor img(x)>
        load_x_tensor = torch.load(image_path)
        load_t_tensor = torch.load(label_path)
        # print("x_tensor:",load_x_tensor,"t_tensor:",load_t_tensor)
        print("label info :",torch.sum(load_t_tensor ,dim=0))
        dataset = TensorDataset(load_x_tensor, load_t_tensor)
        # n_samples = len(dataset)
        # test_size = int(n_samples *0.2)
        # subset_1_indices = list(range(0,test_size))
        # subset_2_indices = list(range(test_size,n_samples))
        # dataset_1 = Subset(dataset,subset_1_indices)
        # dataset_2 = Subset(dataset,subset_2_indices)
        # # _,val_dataset = random_split(test_dataset,)
        test_dataset = DataLoader(
            dataset, batch_size=8, generator=torch.Generator('cpu'), shuffle=False)
        # print(n_samples)
        # print(load_x_tensor.shape)
        # del load_x_tensor,load_t_tensor
        # for x_test,t_label_test in test_dataset:
                # if i == random_train:
                # x_train,t_label_train = train_dataset
                # x_train.to(self.device, non_blocking=True)
                # t_label_train.to(self.device, non_blocking=True)
            # x_test = x_test.to(self.device,non_blocking=True)
            # t_label_test = t_label_test.to(self.device, non_blocking=True)
        for x_test,t_label_test in test_dataset:
                # if i == random_train:
                # x_train,t_label_train = train_dataset
                # x_train.to(self.device, non_blocking=True)
                # t_label_train.to(self.device, non_blocking=True)
                x_test = x_test.to(self.device,non_blocking=True)
                t_label_test = t_label_test.to(self.device, non_blocking=True)
                self.intersection_test = self.net(x_test)
                accuracy += multiclass_accuracy(
                    input=torch.max(self.intersection_test,1)[1],
                    target=torch.max(t_label_test,1)[1],
                    num_classes=8,
                    average="micro"
                ).item()

        print("model_accuracy:",accuracy*100/len(test_dataset))

        return accuracy

    def save_bagfile(self, dataset_tensor,save_path,file_name):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(dataset_tensor, path+file_name)
        print("save_path_path!!: ",save_path + file_name)

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
    b2t = bag_to_tensor()
