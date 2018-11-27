
import torch
import torch.nn as nn

cfg = {
       'CNN16':[17,16,3,1],
       'CNN32':[17,32,3,1],   
       'CNN64':[17,64,3,1],
       'CNN128':[17,128,3,1],
       'CNN32_5':[17,32,5,2],
       'CNN64_5':[17,64,5,2],
       'CNN128_5':[17,128,5,2],
       }
class deepcnn(nn.Module):
    def __init__(self, name):
        super(deepcnn, self).__init__()
        self.dncnn = self._make_dncnn(cfg[name])

    def forward(self, x):
        out = self.dncnn(x)
        return x-out
    
    def _make_dncnn(self, cfg):
        layers = []
        depth = cfg[0] # depth of network
        ch = cfg[1] # number of channels
        k = cfg[2] # kernel size
        padding = cfg[3]
        layers += [nn.Conv2d(3,ch,k,padding=padding),
                  nn.ReLU(True)]
        for i in range(depth-2):
            layers += [nn.Conv2d(ch,ch,k,padding=padding),
#                       nn.Dropout2d(p=0.9),
                       nn.BatchNorm2d(ch,eps=0.0001, momentum = 0.95),
                       nn.ReLU(True)]
        layers += [nn.Conv2d(ch,3,k,padding=padding)]
        return nn.Sequential(*layers)
    
def test():
    net = deepcnn('CNN32_5')
    x = torch.randn(2,3,96,96)
    y = net(x)
    print(y.size())
#test()