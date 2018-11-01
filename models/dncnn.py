
import torch
import torch.nn as nn

cfg = {
       'CNN16':[17,16,3],
       'CNN32':[17,32,3],        
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
        layers += [nn.Conv2d(3,ch,k,padding=1),
                  nn.ReLU(True)]
        for i in range(depth-2):
            layers += [nn.Conv2d(ch,ch,k,padding=1),
                       nn.BatchNorm2d(ch,eps=0.0001, momentum = 0.95),
                       nn.ReLU(True)]
        layers += [nn.Conv2d(ch,3,k,padding=1)]
        return nn.Sequential(*layers)
    
def test():
    net = deepcnn('CNN17')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
#test()