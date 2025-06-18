import torch
from torch import nn
import math


class LNCNN(nn.Module):
    def __init__(self, in_channels, dropout):
        super(CNN, self).__init__()

        self.cn1 = nn.Conv3d(in_channels, 64, kernel_size=(5,5,5), stride=(1,1,1), padding=0, bias=False)
        self.cn2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn3 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn4 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn5 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn6 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn7 = nn.Conv3d(64, 64, kernel_size=(2,2,2), stride=(1,1,1), padding=0, bias=False)
        self.cn8 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn9 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=False)
        self.cn10 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding=1, bias=False)
        self.cn11 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding=1, bias=False)
        self.cn12 = nn.Conv3d(64, 32, kernel_size=(3,3,1), stride=(1,1,1), padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(64)
        self.bn8 = nn.BatchNorm3d(64)
        self.bn9 = nn.BatchNorm3d(64)
        self.bn10 = nn.BatchNorm3d(64)
        self.bn11 = nn.BatchNorm3d(64)
        self.bn12 = nn.BatchNorm3d(32)

        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        #self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.linear = nn.LazyLinear(out_features=256)
        self.classify = nn.Linear(256, 1)
        self.relu = nn.ReLU()


    def forward(self, x, edge_index, edge_attr, batch, clinical, radiomics=None):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.cn4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.cn5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.cn6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.cn7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.cn8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.cn9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.cn10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.cn11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.cn12(x)
        x = self.bn12(x)
        x = self.relu(x)
        #x = self.avgpool(x)
        x = self.flatten(x)
        x = torch.cat((x, clinical), 1)
         
        x = self.linear(x)
        x = self.dropout(x)

        x = self.classify(x)

        return x.squeeze()

    
       
