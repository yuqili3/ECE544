from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import pickle
from PIL import Image

dataDir = '../cifar'
outDir = '../cifar'
#dataDir = '/home/yuqi/spinner/dataset/cifar10'
#outDir = dataDir
def add_noise_and_save(dataDir, outDir, sigma,num_copy = 3):
    trainset = torchvision.datasets.CIFAR10(root=dataDir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=dataDir, train=False, download=True)
    
    train_data = trainset.train_data/255.0
    N = len(train_data)
    train_data_noisy = np.zeros((num_copy*N, 32, 32, 3))
    for copy in range(num_copy):
        for i in range(N):
            train_data_noisy[i*num_copy + copy] = train_data[i] + np.random.randn(32,32,3)*sigma
    
    test_data = testset.test_data/255.0
    N = len(test_data)
    test_data_noisy = np.zeros((num_copy*N, 32, 32, 3))
    for copy in range(num_copy):
        for i in range(N):
            test_data_noisy[i*num_copy + copy] = test_data[i] + np.random.randn(32,32,3)*sigma
            
#    db = {}
#    db['train_data'] = train_data
#    db['train_labels'] = trainset.train_labels
#    db['test_data'] = test_data
#    db['test_labels'] = testset.test_labels
#    db['train_data_noisy'] = train_data_noisy
#    db['test_data_noisy'] = test_data_noisy
#    db['num_copy'] = num_copy
#    db['sigma'] = sigma
#    fileName = '%s/noisyCifar_sigma%f_copy%d'%(outDir,sigma,num_copy)
#    with open(fileName,'wb') as dbFile:
#        pickle.dump(db, dbFile)
    fileName = '%s/noisyCifar_sigma%f_copy%d'%(outDir,sigma,num_copy)
    np.savez(fileName, 
             train_data=train_data, 
             train_labels=trainset.train_labels, 
             train_data_noisy=train_data_noisy,
             test_data=test_data, 
             test_labels=testset.test_labels, 
             test_data_noisy=test_data_noisy,
             num_copy=num_copy,
             sigma=sigma)

if __name__=='__main__':
    add_noise_and_save(dataDir,outDir,sigma=0.05)


class noisy_cifar10(data.Dataset):
    def __init__(self, sigma, num_copy=3,dataDir='../cifar', train=True, transform=None, target_transform=None):
        self.sigma = sigma
        self.num_copy = num_copy
        self.dataDir=dataDir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        fileName = '%s/noisyCifar_sigma%f_copy%d'%(dataDir,sigma,num_copy)
        if os.path.isfile(fileName):
#            with open(fileName,'rb') as dbFile:
            db = np.load(fileName)
            assert sigma == db['sigma'] and num_copy == db['num_copy']
            if self.train:
                self.train_data = db['train_data']
                self.train_labels = db['train_labels']
                self.train_data_noisy = db['train_data_noisy']
            else:
                self.test_data = db['test_data']
                self.test_labels = db['test_labels']
                self.test_data_noisy = db['test_data_noisy']
        else: 
            raise ValueError('no file found! %s'%(fileName))
    def __len__(self):
        if self.train:
            return len(self.train_data_noisy)
        else:
            return len(self.test_data_noisy)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy_image, image, target) where target is index of the target class.
        """
        idx = int(index//self.num_copy)
        if self.train:
            noisy, img, target = self.train_data_noisy[index], self.train_data[idx], self.train_labels[idx]
        else:
            noisy, img, target = self.test_data_noisy[index], self.test_data[idx], self.test_labels[idx]

        noisy = Image.fromarray(noisy)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            noisy = self.transform(noisy)
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return noisy, img, target
    
    