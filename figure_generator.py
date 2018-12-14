import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dataset
import utils
import torch
from PIL import Image
import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
'''
dataDir = '/home/yuqi/spinner/dataset/stl10'
img_ids = [9,10,11]
sigma_list = [0.05] # , 0.1,  0.2]
sigma = 0.05
plt.figure(figsize = (10, 6))
for i,img_id in enumerate(img_ids):
    testset = dataset.noisy_stl10(sigma, num_train=1,num_test=500, dataDir=dataDir, train=False)
    plt.subplot(3,6,1+i*6)
    plt.axis('off')
    img = testset.test_data[img_id]
    plt.imshow(img)
    plt.title('Noise level:%.2f\nClean'%(sigma))
    
    plt.subplot(3,6,2+i*6)
    plt.axis('off')
    noisy = testset.test_data_noisy[img_id]
    plt.imshow(noisy)
    psnr = utils.PSNR(img - noisy)
    plt.title('Noisy\n PSNR:%.2f'%(psnr))
    plt.xticks
    
    plt.subplot(3,6,3+i*6)
    plt.axis('off')
    testset = dataset.denoised_stl10(sigma,netName='BM3D',num_train=1,dataDir=dataDir,train=False)
    denoised_BM1D = testset.test_data_denoised[img_id]
    plt.imshow(denoised_BM1D)
    psnr = utils.PSNR(img - denoised_BM1D)
    plt.title('BM3D\n PSNR:%.2f'%(psnr))
    
    plt.subplot(3,6,4+i*6)
    plt.axis('off')
    testset = dataset.denoised_stl10(sigma,netName='dncnn_CNN64',num_train=1,dataDir=dataDir,train=False)
    denoised_cnn64  = np.clip(testset.test_data_denoised[img_id],0,1) # in DnCNN it is not clipped
    plt.imshow(denoised_cnn64)
    psnr = utils.PSNR(img - denoised_cnn64)
    plt.title('DnCNN\n PSNR:%.2f'%(psnr))
    
    plt.subplot(3,6,5+i*6)
    plt.axis('off')
    reg_term=0.5
    t7file = '../checkpoints/ckpt_denoise_dncnn_CNN64_classification_sigma%.2f_reg%.2f.t7'%(sigma,reg_term)
    ckpt = torch.load(t7file)
    net = models.dncnn.deepcnn('CNN64').cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(ckpt['net_denoise'])
    img_transform = transforms.Compose([transforms.ToTensor()])
    testset = dataset.noisy_stl10(sigma, num_train=1, num_test=img_id+1, dataDir=dataDir, transform=img_transform,train=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    for batch_idx, (noisy, _, targets) in enumerate(testloader):
        noisy = noisy.cuda()
        denoised_2 = net(noisy)
        if batch_idx == img_id:
            denoised_2 = denoised_2.cpu().detach().numpy().transpose((0, 2, 3, 1))[0]
            denoised_joint = np.clip(denoised_2,0,1)
    plt.imshow(denoised_joint)
    psnr = utils.PSNR(denoised_joint - img)
    plt.title('Fine Tune\n PSNR:%.2f'%(psnr))
    
    plt.subplot(3,6,6+i*6)
    plt.axis('off')
    reg_term=0.5
    t7file = '../checkpoints/ckpt_denoise_random_init_dncnn_CNN64_classification_sigma%.2f_reg%.2f.t7'%(sigma,reg_term)
    ckpt = torch.load(t7file)
    net = models.dncnn.deepcnn('CNN64').cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(ckpt['net_denoise'])
    img_transform = transforms.Compose([transforms.ToTensor()])
    testset = dataset.noisy_stl10(sigma, num_train=1, num_test=img_id+1, dataDir=dataDir, transform=img_transform,train=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    for batch_idx, (noisy, _, targets) in enumerate(testloader):
        noisy = noisy.cuda()
        denoised_2 = net(noisy)
        if batch_idx == img_id:
            denoised_2 = denoised_2.cpu().detach().numpy().transpose((0, 2, 3, 1))[0]
            denoised_joint = np.clip(denoised_2,0,1)

    denoised_joint = np.clip(denoised_joint,0,1)
    plt.imshow(denoised_joint)
    psnr = utils.PSNR(denoised_joint - img)
    plt.title('Joint\n PSNR:%.2f'%(psnr))
    
    # STILL NEED A IMAGE OF MIDDLE IMAGE IN JOINT LEARNING
plt.savefig('Figures/image_display_sigma%.2f_%d.png'%(sigma, img_id),dpi=400)
'''

sigma_list= [0.05,0.1,0.2]
reg_list=[0,0.1,0.2,0.5,0.7,0.9]
accu = [[93.88, 94.48, 93.86, 93.86, 94.16, 94.22],
        [91.56,91.48,91.00,91.72, 91.96, 91.94],
        [86.52, 85.68,86.44, 86.24, 86.62, 86.26]]
accu_fixed=[93.6, 88.62, 76.84]
plt.figure()
for i,sigma in enumerate(sigma_list):
    plt.plot(reg_list, accu[i],'--o',label='$\sigma$=%.2f'%(sigma),linewidth=4)
    plt.plot(reg_list, accu_fixed[i]*np.ones(len(reg_list)),':',label='$\sigma$=%.2f,DnCNN'%(sigma),linewidth=3)
    plt.legend(loc='best')
    plt.xlabel('Rrgularization Weight $\\alpha$')
    plt.grid()
    plt.ylabel('Classification Accuracy (%)')


