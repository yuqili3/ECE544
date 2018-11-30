import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import dataset
import utils
import models
import matplotlib.pyplot as plt


def show_result(test_image, noisy_image, output_image1, output_image2, count):
    path = 'outputs/' + str(count) + '.png'
    # Actual Images
    # size_figure_grid = 2
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    test_image = test_image.cpu().detach().numpy().transpose((0, 2, 3, 1))
    # for k in range(1):
    ax[0, 0].cla()
    ax[0, 0].imshow(test_image[0])

    noisy_image = noisy_image.cpu().detach().numpy().transpose((0, 2, 3, 1))
    # for k in range(1):
    ax[0, 1].cla()
    ax[0, 1].imshow(noisy_image[0])

    output_image1 = output_image1.cpu().detach().numpy().transpose((0, 2, 3, 1))
    ax[1, 0].cla()
    ax[1, 0].imshow(np.clip(output_image1[0], 0, 1))

    output_image2 = output_image2.cpu().detach().numpy().transpose((0, 2, 3, 1))
    ax[1, 1].cla()
    ax[1, 1].imshow(np.clip(output_image2[0], 0, 1))

    # label = 'Original Image'
    # fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()


parser = argparse.ArgumentParser(description='Output images using independent and jointly trained denoiser')
parser.add_argument('--copy', default=1, type=int, help='number of noisy image copies')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level sigma')
parser.add_argument('--reg_term', default=0.5, type=float, help='lambda:parameters distributed across two losses')
parser.add_argument('--netType',default='CNN64',type=str, help='type of denoising CNN')
args = parser.parse_args()

sigma = args.sigma
num_copy = args.copy
reg_term = args.reg_term
netType = args.netType
netName = 'dncnn_%s'%(netType)

# Independent denoiser loading
checkpoint = torch.load('./checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName,sigma,num_copy))
net_indep_denoiser = models.dncnn.deepcnn(netName[6:]).cuda()
net_indep_denoiser = torch.nn.DataParallel(net_indep_denoiser)
net_indep_denoiser.load_state_dict(checkpoint['net'])


# Denoiser with joint training
checkpoint = torch.load('./checkpoints/ckpt_denoise_random_init_%s_classification_sigma%.2f_reg%.2f.t7'%(netName,sigma,reg_term))
net_joint_denoiser = models.dncnn.deepcnn(netName[6:]).cuda()
net_joint_denoiser = torch.nn.DataParallel(net_joint_denoiser)
net_joint_denoiser.load_state_dict(checkpoint['net_denoise'])


# Dataloader part
num_train = 2000  # max 5000
num_test = 500  # max 8000
dataDir = '/home/yuqi/spinner/dataset/stl10'

img_transform = transforms.Compose([transforms.ToTensor()])
testset = dataset.noisy_stl10(sigma, num_train=num_train, num_test=num_test,num_copy=num_copy, dataDir=dataDir, transform=img_transform,train=False)
testloader = DataLoader(testset, batch_size=1, shuffle=True)


for batch_idx, (noisy, img, targets) in enumerate(testloader):
    noisy, img = noisy.cuda(), img.cuda()

    denoised_1 = net_indep_denoiser(noisy)
    denoised_2 = net_joint_denoiser(noisy)
    show_result(img, noisy, denoised_1, denoised_2, 1)
    break



