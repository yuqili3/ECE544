import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import utils
            
class noisy_cifar10(data.Dataset):
    def __init__(self, sigma, num_copy=1,num_train=5000, num_test=500,dataDir='../cifar', train=True, transform=None, target_transform=None):
        self.sigma = sigma
        self.num_copy = num_copy
        self.dataDir=dataDir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        fileName = '%s/noisyCifar_sigma%.2f_copy%d.npz'%(dataDir,sigma,num_copy)
        if not os.path.isfile(fileName):
            print('no file found! %s: generating:...'%(fileName))
            utils.add_noise_and_save(dataDir=dataDir,outDir=dataDir,sigma=sigma,num_copy=num_copy)
            print('%s File Generated'%(fileName))
        
        db = np.load(fileName)
        assert sigma == db['sigma'] and num_copy == db['num_copy']
        if self.train:
            self.train_data = db['train_data'][:num_train]
            self.train_labels = db['train_labels'][:num_train]
            self.train_data_noisy = db['train_data_noisy'][:num_train]
        else:
            self.test_data = db['test_data'][:num_test]
            self.test_labels = db['test_labels'][:num_test]
            self.test_data_noisy = db['test_data_noisy'][:num_test]
            
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
    
class noisy_stl10(noisy_cifar10):
    def __init__(self, sigma, num_copy=1,num_train=2000, num_test=500,dataDir='../stl10', train=True, transform=None, target_transform=None):
        self.sigma = sigma
        self.num_copy = num_copy
        self.dataDir=dataDir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        fileName = '%s/noisySTL10_sigma%.2f_copy%d.npz'%(dataDir,sigma,num_copy)
        if not os.path.isfile(fileName):
            print('no file found! %s: generating:...'%(fileName))
            utils.add_noise_and_save(dataDir=dataDir,outDir=dataDir,sigma=sigma,num_copy=num_copy)
            print('%s File Generated'%(fileName))
            
        db = np.load(fileName)
        assert sigma == db['sigma'] and num_copy == db['num_copy']
        if self.train:
            self.train_data = np.transpose(db['train_data'][:num_train],(0,2,3,1)) # previously N*3*96*96, now N*96*96*3
            self.train_data_noisy = np.transpose(db['train_data_noisy'][:num_train],(0,2,3,1))
            self.train_labels = db['train_labels'][:num_train]
        else:
            self.test_data = np.transpose(db['test_data'][:num_test],(0,2,3,1))
            self.test_data_noisy = np.transpose(db['test_data_noisy'][:num_test],(0,2,3,1))
            self.test_labels = db['test_labels'][:num_test]

class denoised_stl10(noisy_cifar10):
    def __init__(self,sigma,netName, num_copy=1,num_train=2000, num_test=500,dataDir='../stl10', train=True, transform=None, target_transform=None):
        self.sigma = sigma
        self.num_copy = num_copy
        self.dataDir=dataDir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        fileName = '%s/denoisedSTL10_%s_sigma%.2f_copy%d.npz'%(dataDir,netName,sigma,num_copy)
        if not os.path.isfile(fileName):
            print('no file found! %s: generating:...'%(fileName))
            utils.get_denoised_dataset(dataDir=dataDir,outDir=dataDir,sigma=sigma,netName=netName, num_copy=num_copy)
            print('%s File Generated'%(fileName))
            
        db = np.load(fileName)
        assert sigma == db['sigma'] and num_copy == db['num_copy']
#        if self.train:
#            self.train_data = np.transpose(db['train_data'][:num_train],(0,2,3,1))
#            self.train_data_denoised = np.transpose(db['train_data_denoised'][:num_train],(0,2,3,1))
#            self.train_labels = db['train_labels'][:num_train]
#        else:
#            self.test_data = np.transpose(db['test_data'][:num_test],(0,2,3,1))
#            self.test_data_denoised = np.transpose(db['test_data_denoised'][:num_test],(0,2,3,1))
#            self.test_labels = db['test_labels'][:num_test]
        if self.train:
            self.train_data = db['train_data'][:num_train]
            self.train_data_denoised = db['train_data_denoised'][:num_train]
            self.train_labels = db['train_labels'][:num_train]
        else:
            self.test_data = db['test_data'][:num_test]
            self.test_data_denoised = db['test_data_denoised'][:num_test]
            self.test_labels = db['test_labels'][:num_test]

if __name__ == '__main__':
    trainset = denoised_stl10(sigma=0.1,netName='dncnn_CNN32',dataDir='../stl10',transform=transforms.ToTensor())