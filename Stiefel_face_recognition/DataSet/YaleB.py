from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
#from PIL import Image

import os
import numpy as np
import gzip
import pickle

from collections import defaultdict

#from sklearn.decomposition import PCA


class YaleB(data.Dataset):
    def __init__(self, filepath=None,train=True):


        self.filepath=filepath

        self.train_data=np.load(filepath+'/YaleB_train_3232.npy')
        self.train_labels=np.load(filepath+'/YaleB_train_gnd.npy')
        self.train_labels_num=np.load(filepath+'/YaleB_train_gnd.npy')
        self.test_data=np.load(filepath+'/YaleB_test_3232.npy')
        self.test_labels=np.load(filepath+'/YaleB_test_gnd.npy')
        self.test_labels_num=np.load(filepath+'/YaleB_test_gnd.npy')

        self.train_labels=self.train_labels.astype(np.long)-1
        self.test_labels=self.test_labels.astype(np.long)-1
        self.train_labels_num=self.train_labels_num.reshape(self.train_labels_num.shape[0])
        self.test_labels_num=self.test_labels_num.reshape(self.test_labels_num.shape[0])



        if train==True:
            self.data=self.train_data
            self.labels=self.train_labels
            self.labels_num=self.train_labels_num.tolist()
        else:
            self.data=self.test_data
            self.labels=self.test_labels
            self.labels_num=self.test_labels_num.tolist()

        self.data=self.data.astype(np.float32)

        Index = defaultdict(list)
        for i, label in enumerate(self.labels_num):
            Index[label].append(i)

        self.Index = Index

        classes = list(set(self.labels_num))
        self.classes = classes
    

    def __getitem__(self, index):

        img, label = self.data[index], self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels_num)






