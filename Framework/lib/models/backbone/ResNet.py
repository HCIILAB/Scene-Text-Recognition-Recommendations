import torch.nn as nn
import math
import torch

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size =1,stride =stride,bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride) # 下采样发生在conv3x3中，一种提升准确率的trick
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, strides, compress_layer=False,in_channel=1, multiscale=[]):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=strides[0], padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 32, layers[0],stride=strides[1])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4])        
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5])

        self.compress_layer = compress_layer   
        self.multiscale = multiscale # to keep some feature maps

        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1),bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))
        self.out_planes = 256 if compress_layer else 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_features = []
        x = self.layer0(x)
        if 0 in self.multiscale:
            out_features.append(x)
        x = self.layer1(x)
        if 1 in self.multiscale:
            out_features.append(x)
        x = self.layer2(x)
        if 2 in self.multiscale:
            out_features.append(x)
        x = self.layer3(x)
        if 3 in self.multiscale:
            out_features.append(x)
        x = self.layer4(x)
        if 4 in self.multiscale:
            out_features.append(x)  
        x = self.layer5(x)
        if 5 in self.multiscale:
            out_features.append(x)
        # 是否有压缩层
        if self.compress_layer:
            x = self.layer6(x)
        if 6 in self.multiscale:
            out_features.append(x)

        if len(out_features) != 0:
            return out_features
        else:
            return x

def ResNet45(strides, compress_layer=False,in_channel=1, multiscale=[],get_mask=False):
    model = ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer, in_channel,multiscale)
    return model

if __name__=="__main__":
    #1D:
    strides_1D = [(1,1),(2,2),(2,2),(2,1),(2,1),(2,1)]
    model_1D = ResNet45(strides_1D,in_channel=3)
    #2D:
    strides_2D = [(1,1),(2,2),(1,1),(2,2),(1,1),(1,1)]
    model_2D = ResNet45(strides_2D,in_channel=3).to('cuda')
    #multiscale
    # model_multiscale = ResNet45(strides_1D,in_channel=3,multiscale=[1,2,3,4,5])
    # #IAM:
    # strides_IAM = [(2,1),(2,2),(2,2),(2,1),(2,2),(2,2)]
    # model_IAM = ResNet45(strides_IAM,compress_layer=True,in_channel=3)
    # x_scene = torch.rand(10,3,32,128)
    # x_IAM = torch.rand(10,3,192,2048)
    # print('1D shape:',model_1D(x_scene).shape)
    # print('2D shape:',model_2D(x_scene).shape)
    # print('1D multisale shape:',[fe.shape for fe in model_multiscale(x_scene)])
    # print('IAM shape:',model_IAM(x_IAM).shape)
    input_images = torch.rand(256,3,32,128).to('cuda')
    print(model_2D(input_images).shape)
