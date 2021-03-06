from jittor import nn
import jittor as jt
from lib.utils import MODELS, build_from_cfg
import numpy as np

@MODELS.register_module()
class PMG(nn.Module):
    def __init__(self, model, feature_size, num_classes):
        super(PMG, self).__init__()

        self.features = build_from_cfg(model, MODELS, out_stages=[1, 2, 3, 4, 5])
        #self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU()

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.Linear(feature_size, num_classes),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.Linear(feature_size, num_classes),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.Linear(feature_size, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(),
            nn.Linear(feature_size, num_classes),
        )

    def execute_train(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)
        
        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)
          
        x_concat = jt.concat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)

        return xc1, xc2, xc3, x_concat


    def execute_test(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)
        
        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
          
        x_concat = jt.concat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)

        return xc1, xc2, xc3, x_concat

    def params_dict(self):
        lst = [
            {'name': 'classifier_concat', 'params': self.classifier_concat.parameters()}, 
            {'name': 'conv_block1', 'params': self.conv_block1.parameters()},
            {'name': 'classifier1', 'params': self.classifier1.parameters()},
            {'name': 'conv_block2', 'params': self.conv_block2.parameters()},
            {'name': 'classifier2', 'params': self.classifier2.parameters()},
            {'name': 'conv_block3', 'params': self.conv_block3.parameters()},
            {'name': 'classifier3', 'params': self.classifier3.parameters()},
            {'name': 'features', 'params': self.features.parameters()}
            ]
        return lst

    def execute(self, x):
        if self.is_training():
            return self.execute_train(x)
        else: 
            return self.execute_test(x)
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def execute(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
