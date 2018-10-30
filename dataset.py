import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import utils

class noisy_cifar10(data.Dataset):
    def __init__(self, sigma, num_copy=3,dataDir='../cifar', train=True, transform=None, target_transform=None):
        self.sigma = sigma
        self.num_copy = num_copy
        self.dataDir=dataDir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        fileName = '%s/noisyCifar_sigma%.2f_copy%d.npz'%(dataDir,sigma,num_copy)
        if os.path.isfile(fileName):
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
            print('no file found! %s: generating:...'%(fileName))
            utils.add_noise_and_save(dataDir=dataDir,outDir=dataDir,sigma=sigma,num_copy=num_copy)
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
            noisy = self.train_data_noisy[index]
            img, target = self.train_data[idx], self.train_labels[idx]  
        else:
            noisy = self.test_data_noisy[index]
            img, target = self.test_data[idx], self.test_labels[idx]
        
        noisy = (noisy*255).astype(np.uint8)
        img = (img*255).astype(np.uint8)
        # Image.fromarray only defined for uint8
        noisy = Image.fromarray(noisy)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            noisy = self.transform(noisy)
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return noisy, img, target