# This script is for jointly training denoising and classification
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.utils as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import dataset
import argparse

import models

parser = argparse.ArgumentParser(description='Multi-task learning: Classification and Denoising')
parser.add_argument('--epoch', default=30, type=int, help='number of training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level sigma')
parser.add_argument('--reg_term', default=0.5, type=int, help='lambda:parameters distributed across two losses')

args = parser.parse_args()
# Classification Model Pre-loading
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# }


def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        # model.load_state_dict(utils.model_zoo.load_url(model_urls['resnet18'], model_dir='./'))
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(torch.load(classification_model_path))
    return model


classification_model_path = '/home/ankit/courses/pattern_recognition/ECE544/clean_classifier_model.pkl'
classification_model = resnet18()
classification_model.cuda()
# Learning rate for different parameters
ignored_params = list(map(id, classification_model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     classification_model.parameters())

learning_rate = 1e-5
optimizer_classifier = optim.Adam([
    {'params': base_params},
    {'params': classification_model.fc.parameters(), 'lr': 1e-3}
], lr=learning_rate)

criterion_CE = nn.CrossEntropyLoss()

sigma = args.sigma
num_copy = 1

# Denoising model
netType = 'CNN64'
netName = 'dncnn_%s'%(netType)
net_denoise = models.dncnn.deepcnn(netType).cuda()
denoising_model_path = torch.load('./checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName, sigma, num_copy))
# print(denoising_model_path)
net_denoise = torch.nn.DataParallel(net_denoise)
net_denoise.load_state_dict(denoising_model_path['net'])
net_denoise.cuda()


criterion_MSE = nn.MSELoss()
lr = 1e-4
optimizer_denoising = optim.Adam(net_denoise.parameters(), lr=lr, weight_decay=1e-5)

# Fixed Transform
fixed_transform = models.helper.transform().cuda()

# Data Part
dataDir = '/home/yuqi/spinner/dataset/stl10'
num_train = 2000   # max 5000
num_test = 500
batch_size = 32

img_transform = transforms.Compose([transforms.ToTensor()])

trainset = dataset.noisy_stl10(sigma, num_train=num_train, num_test=num_test,num_copy=num_copy, dataDir=dataDir, transform=img_transform, train=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = dataset.noisy_stl10(sigma, num_train=num_train, num_test=num_test,num_copy=num_copy, dataDir=dataDir, transform=img_transform, train=False)
testloader = DataLoader(testset, batch_size=20, shuffle=True)


num_epochs = args.epoch
reg_term = args.reg_term

for epoch in range(num_epochs):
    train_accuracy = []
    for batch_idx, (noisy, clean, targets) in enumerate(trainloader):
        noisy, clean, targets = Variable(noisy.cuda()), Variable(clean.cuda()), Variable(targets.cuda())
        optimizer_classifier.zero_grad()
        optimizer_denoising.zero_grad()
        # print("1", noisy.size())
        denoised_output = net_denoise(noisy)
        # print("2", denoised_output.size())
        transformed_denoised = fixed_transform(denoised_output)
        # print("3", transformed_denoised.size())
        class_output = classification_model(transformed_denoised)

        loss_ce = criterion_CE(class_output, targets)
        loss_denoise = criterion_MSE(denoised_output, clean)

        net_loss = (1 - reg_term) * loss_ce + reg_term*loss_denoise
        net_loss.backward()
        optimizer_denoising.step()
        optimizer_classifier.step()

        prediction = class_output.data.max(1)[1]
        accuracy = (float(prediction.eq(targets.data).sum()) / float(batch_size)) * 100.0
        train_accuracy.append(accuracy)
    print(epoch, np.mean(train_accuracy))


net_denoise.eval()
classification_model.eval()
test_accuracy = []

for batch_idx, (noisy, clean, targets) in enumerate(testloader):
    noisy, clean, targets = Variable(noisy.cuda()), Variable(clean.cuda()), Variable(targets.cuda())
    optimizer_classifier.zero_grad()
    optimizer_denoising.zero_grad()

    denoised_output = net_denoise(noisy)
    transformed_denoised = fixed_transform(denoised_output)
    class_output = classification_model(transformed_denoised)
    prediction = class_output.data.max(1)[1]
    accuracy = (float(prediction.eq(targets.data).sum()) / float(batch_size)) * 100.0
    test_accuracy.append(accuracy)

print(np.mean(test_accuracy))

torch.save(net_denoise.state_dict(), "./denoising_net.pkl")
torch.save(classification_model.state_dict(), "./classification_net.pkl")
