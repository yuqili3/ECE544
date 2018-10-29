
import torch
import torch.nn as nn

cfg = {
       'MLP2':[[[32*32*3,256],'R',[256,64],'R'],
               [[64,256],'R',[256,32*32*3],'S']],
       'MLP3':[[[32*32*3,1024],'R',[1024,256],'R',[256,64],'R'],
               [[64,256],'R',[1024,256],'R',[256,32*32*3],'S']],
}
class autoencoder(nn.Module):
    def __init__(self, ae_name):
        super(autoencoder, self).__init__()
        self.encoder = self._make_coder(cfg[ae_name][0])
        self.decoder = self._make_coder(cfg[ae_name][1])

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 3,32,32)
        return x
    
    def _make_coder(self, cfg):
        layers = []
        for i,x in enumerate(cfg):
            if x == 'R':
                layers += [nn.ReLU(inplace=True)]
            elif x == 'S':
                layers += [nn.Sigmoid()]
            elif len(x) == 2:
                layers += [nn.Linear(x[0],x[1])]
        return nn.Sequential(*layers)
    
def test():
    net = autoencoder('MLP2')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
#test()