import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dataset
import utils
import torch
from PIL import Image
import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

dataDir = '/home/yuqi/spinner/dataset/stl10'
img_ids = [4,5]
sigma_list = [0.05] # , 0.1,  0.2]
sigma = 0.05
plt.figure(figsize = (10, 6))
for i,img_id in enumerate(img_ids):
    testset = dataset.noisy_stl10(sigma, num_train=1,num_test=500, dataDir=dataDir, train=False)
    plt.subplot(2,5,1+i*5)
    plt.axis('off')
    img = testset.test_data[img_id]
    plt.imshow(img)
    plt.title('Clean\n noise level:%.2f'%(sigma))
    
    plt.subplot(2,5,2+i*5)
    plt.axis('off')
    noisy = testset.test_data_noisy[img_id]
    plt.imshow(noisy)
    psnr = utils.PSNR(img - noisy)
    plt.title('Noisy\n PSNR:%.2f'%(psnr))
    plt.xticks
    
    plt.subplot(2,5,3+i*5)
    plt.axis('off')
    testset = dataset.denoised_stl10(sigma,netName='BM3D',num_train=1,dataDir=dataDir,train=False)
    denoised_BM1D = testset.test_data_denoised[img_id]
    plt.imshow(denoised_BM1D)
    psnr = utils.PSNR(img - denoised_BM1D)
    plt.title('BM3D\n PSNR:%.2f'%(psnr))
    
    plt.subplot(2,5,4+i*5)
    plt.axis('off')
    testset = dataset.denoised_stl10(sigma,netName='dncnn_CNN64',num_train=1,dataDir=dataDir,train=False)
    denoised_cnn65  = np.clip(testset.test_data_denoised[img_id],0,1) # in DnCNN it is not clipped
    plt.imshow(denoised_cnn65)
    psnr = utils.PSNR(img - denoised_cnn65)
    plt.title('DnCNN\n PSNR:%.2f'%(psnr))
    
    plt.subplot(2,5,5+i*5)
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
    plt.imshow(denoised_joint)
    psnr = utils.PSNR(denoised_joint - img)
    plt.title('Joint\n PSNR:%.2f'%(psnr))
    
    # STILL NEED A IMAGE OF MIDDLE IMAGE IN JOINT LEARNING
    plt.savefig('Figures/image_display_sigma%.2f_%d.jpg'%(sigma, img_id),dpi=400)



