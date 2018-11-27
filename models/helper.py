import torch
import torch.nn as nn
import torch.nn.functional as F

outsize=224
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

class transform(nn.Module):
    def __init__(self):
        super(transform, self).__init__()
        self.outsize = 224
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze_(1).unsqueeze_(2).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze_(1).unsqueeze_(2).cuda()

    def forward(self, x):
        up = F.interpolate(x,size=self.outsize,mode='bilinear')
<<<<<<< HEAD
        out = up.sub(self.mean)
        out = out.div(self.std)
=======
        out= up.sub(self.mean.unsqueeze_(1).unsqueeze_(2))
        out= out.div(self.std.unsqueeze_(1).unsqueeze_(2))
>>>>>>> 2a3e5959b461da932b472ca9cb03c70aa98f3e59
        return out
    
def test():
    T = transform()
    x = torch.randn(2,3,96,96)
    y = T(x)
    print(y.size())
# test()
