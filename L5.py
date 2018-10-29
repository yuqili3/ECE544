import torch
from torch import nn
import numpy as np
#t = torch.FloatTensor([1,5,3,7])
#t.requires_grad = True
#s = (t*t).sum()
#s.backward()
#opt = torch.optim.SGD([t],lr = 1e-1)
#opt.step()
##opt.zero_grad()
#print(t.grad)

#t = torch.randn(2,requires_grad = True)
#f = lambda t: 3*t[0]**2 + 1.5*t[1]**2 - 2.3*t[0]*t[1] +5.6*t[0]+0.7*t[1]+1.2
#opt = torch.optim.SGD([t],lr = 1e-1)
#for _ in range(40):
#    opt.zero_grad()
#    loss = f(t)
#    loss.backward()
#    opt.step()
##    print(loss)
#    print(t)

class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
#        self.param = nn.Parameter(torch.ones(4))
        self.param = nn.Parameter(torch.FloatTensor(2)) # randomly init
#        self.param.weight.data.fill_(1)
#        self.param.bias.data.fill_(0)
#        self.seq = nn.Sequential(nn.Linear(4,4), nn.ReLU(), nn.Linear(4,1))
    def forward(self, inp):
#        return (self.param *inp).sum()
#        return self.param(inp)
#        return self.seq(inp)
        p = self.param
        return inp[0]*p[0]**2 + inp[1]*p[1]**2 + inp[2]*p[0]*p[1] +inp[3]*p[0] + inp[4]*p[1] +inp[5]

model = TestModule()
#print(list(model.parameters()))
#inp = torch.FloatTensor([1,2])
inp = torch.FloatTensor([3, 1.5, -2.3, 5.6, 0.7, 1.2])
#obj = model(inp)
opt = torch.optim.SGD(model.parameters(),lr = 1e-1)
#obj.backward()
#print(model.param.grad)

for i in range(0):
    opt.zero_grad()
    obj = model(inp)
    print('\t %d: %f'%(i, obj.item()))
    obj.backward()
    opt.step()
#print(list(model.parameters()))

from torch.utils.data import Dataset, DataLoader
class sampleDataset(Dataset):
    def __init__(self, num_data_points, data, labels):
        self.inputs = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    def __len__(self):  
        return self.inputs.shape[0]
    def __getitem__(self, idx):
        return self.inputs[idx,:], self.labels[idx]
    
#dataset = sampleDataset(10)
#print(len(dataset))
#print(dataset[3])
#dataloader = DataLoader(dataset,batch_size=3,shuffle=True)
#print(len(dataloader))
#tmp = nn.Linear(8,1)
#for i, batch in enumerate(dataloader):
#    print(i, batch[0],batch[1], tmp(batch[0]))

inp, oup = torch.load('linear_regression.data')
#print(inp.shape)
dim = inp.shape[1]
lin_data = sampleDataset(inp.shape[0],inp, oup)
dataloader = DataLoader(lin_data, batch_size=10)
class linearModel(nn.Module):
    def __init__(self, dim):
        super(linearModel, self).__init__()
        self.param = nn.Linear(dim,1)
    def forward(self,inp):
        return self.param(inp)

linearMdl = linearModel(dim)
opt = torch.optim.SGD(linearMdl.parameters(),lr = 1e-4)
for i in range(100):
    for b in dataloader:
        opt.zero_grad()
        pred = linearMdl(b[0]).squeeze()
        loss = nn.MSELoss()(pred, b[1])
        loss.backward()
        opt.step()
print(list(linearMdl.parameters()))
        

