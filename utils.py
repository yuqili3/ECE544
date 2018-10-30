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
import numpy as np
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

    fileName = '%s/noisyCifar_sigma%.2f_copy%d'%(outDir,sigma,num_copy)
    np.savez(fileName, 
             train_data=train_data, 
             train_labels=trainset.train_labels, 
             train_data_noisy=train_data_noisy,
             test_data=test_data, 
             test_labels=testset.test_labels, 
             test_data_noisy=test_data_noisy,
             num_copy=num_copy,
             sigma=sigma)

#if __name__=='__main__':
#    add_noise_and_save(dataDir,outDir,sigma=0.05)
    
def get_output(netName,sigma,num_copy=3,in_img):
    # in_img: 32x32x3 np array, uint8 
    # out_img: 32x32x
    checkpoint = torch.load('../checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName,sigma,num_copy))
    net = dae.autoencoder(netName.split('_')[1]).cuda()
    net.load_state_dict(checkpoint['net'])
    
    in_img = Image.fromarray(in_img.astype(np.uint8))
    in_img = (torch.ToTensor())(in_img).unsqueeze_(0) # now 1x3x32x32
    out_img = net(in_img)
    out_img = numpy.array(torch.squeeze_()).transpose((1,2,0))
    
    return out_img

def PSNR(X):
    s = X.shape
    psnr = 20*np.log10(np.sqrt(s.prod()) * 255 / np.linalg.norm(X))
    return psnr

def denois_example(index):
    

    
    