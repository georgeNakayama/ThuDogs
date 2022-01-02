from jittor import nn
import jittor as jt

def conv1x1(inplanes, outplanes, stride=1, padding=0):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=padding)

def conv3x3(inplanes, outplanes, stride=1, padding=0):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=padding)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1=conv3x3(inplanes, planes, stride=stride, padding=1)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=conv3x3(planes, planes, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.relu=nn.Relu()

        self.downsample = downsample
    
    def execute(self, x):
        identity = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1=conv1x1(inplanes, planes, stride=stride, padding=0)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=conv3x3(planes, planes, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=conv1x1(planes, planes * self.expansion, stride=1, padding=0)
        self.bn3=nn.BatchNorm2d(planes * self.expansion)

        self.relu=nn.Relu()

        self.downsample = downsample
    
    def execute(self, x):
        identity = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)   

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Rnet(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=130):
        super().__init__()
        self.inplanes=64

        self.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.max_pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu=nn.Relu()

        self.layer1=self._make_layer(block, 64, layers[0], stride=1)
        self.layer2=self._make_layer(block, 128, layers[1], stride=2)
        self.layer3=self._make_layer(block, 256, layers[2], stride=2)
        self.layer4=self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax()

    def _make_layer(self, block, planes, num_layers, stride=1):
        network=[]
        downsample = nn.Sequential([nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0), nn.BatchNorm2d(planes * block.expansion)])
        network.append(block(self.inplanes, planes, stride=stride, downsample=downsample))

        self.inplanes = planes * block.expansion
        for _ in range(num_layers):
            network.append(block(self.inplanes, planes, stride=1, downsample=None))

        return nn.Sequential(*network)


    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.max_pool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        out = self.avg_pool(x5)
        out = jt.reshape(out, (out.shape[0],-1))
        out = self.fc(out)
        return out

def _rnet(block, layers, in_channels=3, num_classes=130):
    model = Rnet(block, layers, in_channels=in_channels, num_classes=num_classes)
    return model

def Rnet18():
    return _rnet(BasicBlock, [2, 2, 2, 2])

def Rnet34():
    return _rnet(BasicBlock, [3, 4, 6, 3])

def Rnet50():
    return _rnet(BottleNeck, [3, 4, 6, 3])

def Rnet101():
    return _rnet(BottleNeck, [3, 4, 23, 3])

def Rnet152():
    return _rnet(BottleNeck, [3, 8, 36, 3])

if __name__ == '__main__':
    model = Rnet50()
    x = jt.random([10,3,224,224])
    y = model(x) # [10, 1000]
    print(y.shape)