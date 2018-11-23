import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.utils as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import dataset

# Model Pre-loading
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def resnet18(pretrained=True):
    model = models.resnet.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(utils.model_zoo.load_url(model_urls['resnet18'], model_dir='./'))
    return model


model = resnet18()
model.fc = nn.Linear(512, 10)   # Final layer transform changed as # of outputs is 10
model.cuda()

# Learning rate for different parameters
ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())

learning_rate = 1e-4
optimizer = optim.Adam([
    {'params': base_params},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], lr=learning_rate)

criterion = nn.CrossEntropyLoss()
num_epochs = 10
batch_size = 40

# Transform on training and test data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=224, scale=(0.64, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataDir = '/home/yuqi/spinner/dataset/stl10'
sigma = 0.5
num_copy = 1

# Dataloader for train set
trainset = dataset.noisy_stl10(sigma, num_copy=num_copy, dataDir=dataDir, transform=transform_train, train=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Dataloader for test set
testset = dataset.noisy_stl10(sigma, num_copy=num_copy, dataDir=dataDir, transform=transform_test, train=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

model.train()

for epoch in range(num_epochs):
    train_accuracy = []
    for batch_idx, (inputs, _, targets) in enumerate(trainloader):
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prediction = outputs.data.max(1)[1]
        accuracy = (float(prediction.eq(targets.data).sum()) / float(batch_size)) * 100.0
        train_accuracy.append(accuracy)
    print(epoch, accuracy)

del inputs, targets

model.eval()
test_acc = []
for batch_idx, (inputs, _, targets) in enumerate(testloader):
    with torch.no_grad():
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        prediction = outputs.data.max(1)[1]
        accuracy = (float(prediction.eq(targets.data).sum()) / float(batch_size))*100.0
        test_acc.append(accuracy)
    del inputs, targets
accuracy_test = np.mean(test_acc)
print("test accuracy", accuracy_test)
