from __future__ import print_function

import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import numpy as np
from scipy.misc import imsave, imread
from PIL import Image
import dataset
import models

# cifar-10 previous
'''
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
    for i in range(N):
        for copy in range(num_copy):
            train_data_noisy[i*num_copy + copy] = np.clip(train_data[i]+np.random.normal(size=train_data[i].shape)*sigma,0.0, 1.0)

    
    test_data = testset.test_data/255.0
    N = len(test_data)
    test_data_noisy = np.zeros((num_copy*N, 32, 32, 3))
    for i in range(N):
        for copy in range(num_copy):    
            test_data_noisy[i*num_copy + copy] = np.clip(test_data[i]+np.random.normal(size=test_data[i].shape)*sigma, 0.0,1.0)

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
'''

#dataDir = outDir = '../stl10'
#dataDir = outDir = '../Dataset/stl10'
def add_noise_and_save(dataDir, outDir, sigma,num_copy = 1):
    trainset = torchvision.datasets.STL10(root=dataDir, split='train', download=True)
    testset = torchvision.datasets.STL10(root=dataDir, split='test', download=True)
    
    train_data = trainset.data/255.0
    train_data_noisy = (np.zeros(train_data.shape) for i in range(num_copy))
    train_data_noisy = np.vstack(train_data_noisy)
    np.random.seed(0) # jsut to make sure we are getting the same noisy image
    for i in range(len(train_data)):
        for copy in range(num_copy):
            train_data_noisy[i*num_copy + copy] = np.clip(train_data[i]+np.random.normal(size=train_data[i].shape)*sigma,0.0, 1.0)

    
    test_data = testset.data/255.0
    test_data_noisy = (np.zeros(test_data.shape) for i in range(num_copy))
    test_data_noisy = np.vstack(test_data_noisy)    
    np.random.seed(1)
    for i in range(len(test_data)):
        for copy in range(num_copy):
            test_data_noisy[i*num_copy + copy] = np.clip(test_data[i]+np.random.normal(size=test_data[i].shape)*sigma, 0.0,1.0)

    fileName = '%s/noisySTL10_sigma%.2f_copy%d'%(outDir,sigma,num_copy)
    np.savez(fileName, 
             train_data=train_data, 
             train_labels=trainset.labels.astype(np.int), 
             train_data_noisy=train_data_noisy,
             test_data=test_data, 
             test_labels=testset.labels.astype(np.int), 
             test_data_noisy=test_data_noisy,
             num_copy=num_copy,
             sigma=sigma)


def get_denoised_dataset(dataDir, outDir, sigma,netName, num_copy = 1):
    checkpoint = torch.load('./checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName,sigma,num_copy))
    net = models.dncnn.deepcnn(netName[6:]).cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])
    device = 'cuda'
    img_transform = transforms.Compose([transforms.ToTensor()])
    
    batch_size_train = 32
    batch_size_test = 20
    trainset = dataset.noisy_stl10(sigma, num_copy=num_copy, dataDir=dataDir, transform=img_transform,train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=False)
    testset = dataset.noisy_stl10(sigma, num_copy=num_copy, dataDir=dataDir, transform=img_transform,train=False)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)
    
    
    train_data_denoised = np.zeros(trainset.train_data.shape)
    test_data_denoised = np.zeros(testset.test_data.shape)
    
    with torch.no_grad():
        img_idx = 0
        for batch_idx, (noisy, img, targets) in enumerate(trainloader):
            noisy, img, targets = noisy.to(device), img.to(device), targets.to(device)
            outputs = net(noisy) # now N*3*96*96
            outputs = outputs.cpu().detach().numpy().transpose((0,2,3,1)) # now N*96*96*3
            train_data_denoised[batch_idx*batch_size_train:(batch_idx+1)*batch_size_train]=outputs
#            for i in range(len(outputs)):
#                if img_idx%500 == 0:
#                    out_img = np.clip(outputs[i],0,1)
#                    imsave('%s/images/%s_sigma%.2f_train_denoised_%d.jpg'%(outDir,netName,sigma,img_idx),out_img)
#                img_idx += 1
    
    with torch.no_grad():
        img_idx = 0
        for batch_idx, (noisy, img, targets) in enumerate(testloader):
            noisy, img, targets = noisy.to(device), img.to(device), targets.to(device)
            outputs = net(noisy) # now N*3*96*96
            outputs = outputs.cpu().detach().numpy().transpose((0,2,3,1)) # now N*96*96*3
            test_data_denoised[batch_idx*batch_size_test:(batch_idx+1)*batch_size_test]=outputs
            for i in range(len(outputs)):
                if img_idx%100 == 0:
                    out_img = np.clip(outputs[i],0,1)
                    imsave('%s/images/%s_sigma%.2f_test_denoised_%d.jpg'%(outDir,netName,sigma,img_idx),out_img)
                    img_clean=img.cpu().detach().numpy().transpose((0,2,3,1))[i]
                    imsave('%s/images/%s_sigma%.2f_test_img_%d.jpg'%(outDir,netName,sigma,img_idx),img_clean)
                    noisy=noisy.cpu().detach().numpy().transpose((0,2,3,1))[i]
                    imsave('%s/images/%s_sigma%.2f_test_noisy_%d.jpg'%(outDir,netName,sigma,img_idx),noisy)
                img_idx += 1
                
    fileName = '%s/denoisedSTL10_%s_sigma%.2f_copy%d.npz'%(dataDir,netName,sigma,num_copy)
    np.savez(fileName, 
             train_data_denoised=train_data_denoised, 
             train_labels=trainset.train_labels, 
             train_data=trainset.train_data,
             train_data_noisy=trainset.train_data_noisy,
             test_data_denoised=test_data_denoised, 
             test_labels=testset.test_labels, 
             test_data=testset.test_data,
             test_data_noisy=testset.test_data_noisy,
             num_copy=num_copy,
             sigma=sigma,
             netName=netName)
    
def get_output(in_img,netName,sigma=0.05,num_copy=1):
    # in_img: 32x32x3 np array, uint8 
    # out_img: 32x32x3 np array float 32
    checkpoint = torch.load('../checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName,sigma,num_copy))
#    net = models.dae.autoencoder(netName.split('_')[1]).cuda()
    net = models.dncnn.deepcnn(netName[6:]).cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])
    
    if np.max(in_img) < 2:
        in_img *= 255.0
    in_img = Image.fromarray(in_img.astype(np.uint8))
    in_img = (transforms.ToTensor()(in_img)).unsqueeze_(0) # now 1x3x32x32
    out_img = net(in_img)
    out_img = out_img.cpu().squeeze_().detach().numpy().transpose((1,2,0))
    out_img = np.clip(out_img,0,1)
    return out_img

def PSNR(X):
    s = np.array(X.shape)
    psnr = 20*np.log10(np.sqrt(s.prod()) / np.linalg.norm(X))
    return psnr

def denois_example(index,netName='dncnn_CNN16',sigma=0.1,num_copy=1,dataDir='/home/yuqi/spinner/dataset/stl10'):
    testset = dataset.noisy_stl10(sigma, num_copy=num_copy, dataDir=dataDir,train=False)
    noisy = testset.test_data_noisy[index] # range [0,1]
    img = testset.test_data[int(index//num_copy)] # range [0,1]
    denoised = get_output(in_img=noisy, netName=netName, sigma=sigma, num_copy=num_copy)
    psnr = PSNR((img - denoised))
    print(np.max(img),np.min(img))
    print(np.max(denoised),np.min(denoised))
    imsave('../result/test_noisy_%d.jpg'%(index),noisy)
    imsave('../result/test_img_%d.jpg'%(index),img)
    imsave('../result/test_denoised_%d.jpg'%(index),denoised)
    return psnr

if __name__ == '__main__':
    psnr = denois_example(5, num_copy=1)
    print(psnr)
    