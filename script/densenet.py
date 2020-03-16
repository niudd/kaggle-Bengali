import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class DenseNet(nn.Module):
    
    def __init__(self, num_classes=186, debug=False):
        """
        features dimension:
        
        """
        super(DenseNet, self).__init__()
        
        self.debug = debug

        self.backbone = torchvision.models.densenet121(pretrained=True).features

        ##use generalized mean pooling instead of nn.AdaptiveAvgPool2d
        self.avg_pool = GeM()
        #self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        #self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes, bias=True),
            nn.BatchNorm1d(num_classes),
        )

    def logits(self, x):
        x_avg = self.avg_pool(x)
        x_avg = x_avg.view(x_avg.size(0), -1)
        if self.debug:
            print("avg_pool: ", x_avg.size())
        x_avg = self.dropout(x_avg)
        
        x = self.fc(x_avg)
        return x

    def forward(self, x):
        #resize inputs
        #x = F.interpolate(x, (224,224))

        #(N,1,224,224)-->(N,3,224,224)
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.backbone(x)
        if self.debug:
            print('extract_features: ', x.size())
        x = self.logits(x)
        if self.debug:
            print('logits: ', x.size())
        return x

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.parameter.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'




