import os, sys
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import scipy.optimize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# print(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .autoencoder.model import *
from .autoencoder import data, track

class FSEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.pool = FSPool(dim, 20, relaxed=kwargs.get('relaxed', True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x, perm = self.pool(x, n_points)
        
        
        #x=nn.Dropout(p=0.2)(x)
        x = self.lin(x)
        return x, perm

class FSEncoder_set(nn.Module):
    def __init__(self, dim, point_dim = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_channels * set_size, dim)
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.pool = FSPool(dim, 20, relaxed=kwargs.get('relaxed', True))

    def forward(self, x, n_points, *args):
        x = x.view(x.size(0), -1)
        x = self.enc(x)
        x, perm = self.pool(x, n_points)
        x = self.lin(x)
        return x



class SumEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.sum(2)
        x = self.lin(x)
        return x,x.clone()


class MaxEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.max(2)[0]
        x = self.lin(x)
        return x,x.clone()


class MeanEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.sum(2) / n_points.unsqueeze(1).float()
        x = self.lin(x)
        return x,x.clone()

class BLP(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()

        FSE=MeanEncoder
        self.enc_U = FSE(input_channels = input_channels,
                                output_channels = output_channels,
                                set_size = kwargs['set_size_U'],
                               dim = dim,
                                **kwargs)
        
        
        self.enc_V = FSE(input_channels = input_channels,
                                output_channels = output_channels,
                                set_size = kwargs['set_size_V'],
                               dim = dim,
                                **kwargs)
        self.classifier = nn.Linear(2*output_channels, 1)
    def forward(self, sample, *args):
        U, V, n_points_U, n_points_V = sample
        x_U, _ = self.enc_U(U, n_points_U)
        x_V, _ = self.enc_V(V, n_points_V)
        x = self.classifier(torch.cat([x_U, x_V], dim=1))
        return x

class BLP_(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()

        FSE=FSEncoder
        self.enc_U = FSE(input_channels = input_channels,
                                output_channels = output_channels,
                                set_size = kwargs['set_size_U'],
                               dim = dim,
                                **kwargs)
        
        
        # self.enc_V = FSE(input_channels = input_channels,
        #                         output_channels = output_channels,
        #                         set_size = kwargs['set_size_V'],
        #                        dim = dim,
        #                         **kwargs)
        self.classifier = nn.Linear(output_channels, 1)
    def forward(self, sample, *args):
        U, V, n_points_U, n_points_V = sample
        # pdb.set_trace()

        U_=torch.cat([U,V],dim=2)

        n_points=n_points_U+n_points_V

        x_U, _ = self.enc_U(U_, n_points)
        # x_V, _ = self.enc_V(V, n_points_V)

        x = self.classifier(x_U)
        return x

class EMB_LAYER(nn.Module):
    def __init__(self,word_map1,word_map2,padd,input_channels, output_channels, dim, set_size_U,set_size_V,**kwargs):
        super().__init__()
        kwargs['set_size_U'] = set_size_U
        kwargs['set_size_V'] = set_size_V
        

        # self.embedding1 = nn.Embedding.from_pretrained(embeddings=word_map1,freeze=False,padding_idx=padd)
        # self.embedding2 = nn.Embedding.from_pretrained(embeddings=word_map2,freeze=False,padding_idx=padd)
        self.embedding1=word_map1
        self.embedding2=word_map2

        self.out_ = BLP_(input_channels = input_channels,
                                output_channels = output_channels,
                               dim = dim,
                                **kwargs)
        self.c_U=set_size_U
        self.c_V=set_size_V
        self.padd=padd

    def forward(self, sample, *args):
        # pdb.set_trace()
        U, V, n_points_U, n_points_V,mask_U,mask_V = sample
        U_=self.embedding1(U)
        V_=self.embedding2(V)

        #print(U_.size())
        
        """mask needed"""

        
        """U_,V_ are tuple types"""
        U_=torch.cat([U_[0],mask_U],dim=2)
        
        V_=torch.cat([V_[0],mask_V],dim=2)



        sample=(torch.transpose(U_,1,2),torch.transpose(V_,1,2),n_points_U,n_points_V)
        x=self.out_(sample)
        return x