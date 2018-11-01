import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.utils import save_image

import argparse
import dataset
import utils
import models

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 denoising AE Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--epoch', default=20, type=int, help='number of training epochs')
parser.add_argument('--copy', default=3, type=int, help='number of noisy image copies')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level sigma')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

dataDir = '../cifar'
lr = args.lr
sigma = args.sigma
num_copy = args.copy

best_MSE = float('inf')  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

img_transform = transforms.Compose([transforms.ToTensor()])

trainset = dataset.noisy_cifar10(sigma, num_copy=num_copy, dataDir=dataDir, transform=img_transform,train=True)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True,num_workers=2)

testset = dataset.noisy_cifar10(sigma, num_copy=num_copy, dataDir=dataDir, transform=img_transform,train=False)
testloader = DataLoader(testset, batch_size=100, shuffle=True,num_workers=2)

def pairwise_potential(img):
    # TODO: implement sqaure pairwise potential
    out=0
    return out

print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#netType = 'CNN1'
#netName = 'dae_%s'%(netType)
#net = models.dae.autoencoder(netType).to(device)
netType = 'CNN16'
netName = 'dncnn_%s'%(netType)
net = models.dncnn.deepcnn(netType).to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(os.path.normpath('../checkpoints')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName,sigma,num_copy))
    net.load_state_dict(checkpoint['net'])
    best_MSE = checkpoint['mse']
    start_epoch = checkpoint['epoch']


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[60,120,180], gamma=0.2)


def train(epoch):
    print('\nTRAIN: Epoch: %d / %d' %(epoch, args.epoch))
    net.train()
    train_loss = 0
    MSE_loss = 0
    scheduler.step(epoch)
    for batch_idx, (noisy, img, targets) in enumerate(trainloader):
        noisy, img, targets = noisy.to(device), img.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(noisy)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        MSE_loss += nn.MSELoss()(outputs, img).item()

    print('TRAIN: Loss: %.6f | MSE Loss: %.6f'% (train_loss/(batch_idx+1), MSE_loss/(batch_idx+1)))
        
def test(epoch):
    global best_MSE
    net.eval()
    test_loss = 0
    MSE_loss = 0
    with torch.no_grad():
        for batch_idx, (noisy, img, targets) in enumerate(testloader):
            noisy, img, targets = noisy.to(device), img.to(device), targets.to(device)
            outputs = net(noisy)
            loss = criterion(outputs, img)
            
            test_loss += loss.item()
            MSE_loss += nn.MSELoss()(outputs, img).item()
            
    print('\nTest: Epoch: %d / %d' %(epoch, args.epoch))
    print('Test: Loss: %.6f | MSE Loss: %.6f'% (test_loss/(batch_idx+1), MSE_loss/(batch_idx+1)))

    # Save checkpoint.
    MSE = MSE_loss
    if MSE < best_MSE:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'mse': MSE,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.normpath('../checkpoints')):
            os.mkdir(os.path.normpath('../checkpoints'))
        torch.save(state, '../checkpoints/ckpt_%s_sigma%.2f_copy%d.t7'%(netName,sigma,num_copy))
        best_MSE = MSE
        
for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch)
    test(epoch)

