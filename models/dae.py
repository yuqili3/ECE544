
import torch
import torch.nn as nn

cfg = {
       'MLP2':[ [[32*32*3,256],'R',[256,64],'R'],
                [[64,256],'R',[256,32*32*3],'S'] ],
       'CNN1':[ [[3,16,3], [16,16,3],[16,16,3],[16,16,3],'M',[16,16,3],[16,16,3],'M',[16,16,3]],
                [[16,16,3,2,1,1],[16,16,3,1,1],[16,16,3,1,1],[16,16,3,1,1],[16,16,3,1,1],[16,16,3,1,1],[16,3,3,2,1,1]] ],        
       'CNN2':[ [[3,64,3], [64,64,3],[64,128,3],[128,128,3],'M',[128,256,3],[256,256,3],'M',[256,512,3]],
                [[512,256,3,2,1,1],[256,256,3,1,1],[256,128,3,1,1],[128,128,3,1,1],[128,64,3,1,1],[64,64,3,1,1],[64,3,3,2,1,1]] ],        
}
class autoencoder(nn.Module):
    def __init__(self, ae_name):
        super(autoencoder, self).__init__()
        self.encoder = self._make_coder(cfg[ae_name][0])
        self.decoder = self._make_coder(cfg[ae_name][1])

    def forward(self, x):
#        x = x.view(x.size(0),-1)
        x = self.encoder(x)
        x = self.decoder(x)
#        x = x.view(x.size(0),3,32,32)
        return x
    
    def _make_coder(self, cfg):
        layers = []
        for i,x in enumerate(cfg):
            if x == 'R':
                layers += [nn.ReLU(inplace=True)]
            elif x == 'S':
                layers += [nn.Sigmoid()]
            elif x =='M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif len(x) == 2:
                layers += [nn.Linear(x[0],x[1])]
                
            elif len(x) == 3:
                layers += [nn.Conv2d(x[0],x[1],x[2],padding=1)]
                if i < len(cfg)-1:
                    layers += [nn.BatchNorm2d(x[1])] 
                layers += [nn.ReLU(True)]
                
            elif len(x) == 5:
                layers += [nn.ConvTranspose2d(x[0],x[1],x[2],stride=x[3],padding=x[4])]
                if i < len(cfg)-1:
                    layers += [nn.BatchNorm2d(x[1])]
                layers += [nn.ReLU(True)]
                
            elif len(x) == 6:
                layers += [nn.ConvTranspose2d(x[0],x[1],x[2],stride=x[3],padding=x[4],output_padding=x[5])]
                if i < len(cfg)-1:
                    layers += [nn.BatchNorm2d(x[1])]
                layers += [nn.ReLU(True)]
                
        return nn.Sequential(*layers)
    
def test():
    net = autoencoder('CNN1')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
#test()