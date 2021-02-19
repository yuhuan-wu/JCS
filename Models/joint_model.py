import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.utils import ConvBNReLU
from Models.vgg import vgg16
from Models.resnet import resnet18, resnet50, resnet101
from Models.res2net import res2net50_v1b, res2net101_v1b

class FCN(nn.Module):
    def __init__(self, pretrained=None):
        super(FCN, self).__init__()
        self.backbone = resnet18(pretrained)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
       
        self.cls = nn.Conv2d(512, 1, 1, stride=1, padding=0)

    def forward(self, input):
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        saliency_maps = F.interpolate(self.cls(conv5), input.shape[2:], mode='bilinear', align_corners=False)

        return torch.sigmoid(saliency_maps)

class FuseNet(nn.Module):
    def __init__(self, c1=[1,2,3,4,5], c2=[1,2,3,4,5], out_channels=[1,2,3,4,5]):
        super(FuseNet, self).__init__()
        self.cat_modules = nn.ModuleList()
        self.se_modules = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        for i in range(len(c1)):
            self.cat_modules.append(ConvBNReLU(c1[i]+c2[i], out_channels[i]))
            self.se_modules.append(ConvBNReLU(out_channels[i], out_channels[i]))
            self.fuse_modules.append(ConvBNReLU(out_channels[i], out_channels[i]))

    def forward(self, x1, x2):
        x_new = []
        for i in range(5):
            x1[i] = F.interpolate(x1[i], x2[i].shape[2:], mode='bilinear', align_corners=False)
            m = self.cat_modules[i](torch.cat([x1[i], x2[i]], dim=1))
            #print(m.shape)
            m = self.se_modules[i](m)
            #print(m.shape)
            m = self.fuse_modules[i](m)
            #print(m.shape)
            x_new.append(m)
        return x_new[0], x_new[1], x_new[2], x_new[3], x_new[4]



class JCS(nn.Module):
    def __init__(self, pretrained=None, use_carafe=True,
                 enc_channels=[64, 128, 256, 512, 512, 512],
                 dec_channels=[64, 128, 256, 512, 512, 512]):
        super(JCS, self).__init__()
        self.vgg16 = vgg16(pretrained)
        self.res2net = res2net101_v1b("ok")
        self.fuse = FuseNet(c1=[64, 128, 256, 512, 512], c2=[64, 256, 512, 1024, 2048], out_channels=enc_channels[:-1])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gpd = GPD(enc_channels[-1], expansion=4)
        self.gpd1 = GPD(enc_channels[-1], expansion=4, dilation=[1,2,3,4])
        self.fpn = ImprovedDecoder(enc_channels, dec_channels)
        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2,2,0)

    def forward(self, input, res2net_features=None):
        if res2net_features is not None:
            conv1, conv2, conv3, conv4, conv5 = self.vgg16(input)
            conv1r, conv2r, conv3r, conv4r, conv5r = self.res2net(res2net_features)
        conv5 = self.gpd(conv5)
        conv6 = self.pool(conv5)
        conv6 = self.gpd1(conv6)
        conv1, conv2, conv3, conv4, conv5 = self.fuse([conv1, conv2, conv3, conv4, conv5], [conv1r, conv2r, conv3r, conv4r, conv5r])
        features = self.fpn([conv1, conv2, conv3, conv4, conv5, conv6])

        saliency_maps = []
        for idx, feature in enumerate(features[:-1]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )

        return torch.sigmoid(torch.cat(saliency_maps, dim=1))


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels//reduction, bias=True)
        self.linear2 = nn.Linear(in_channels//reduction, in_channels)
        self.act = nn.ReLU(inplace=True)
        

    def forward(self, x):
        N, C, H, W = x.shape
        embedding = x.mean(dim=2).mean(dim=2)
        fc1 = self.act(self.linear1(embedding))
        fc2 = torch.sigmoid(self.linear2(fc1))
        return x * fc2.view(N, C, 1, 1)
    


class GPD(nn.Module):
    def __init__(self, in_channels, expansion=4, dilation=[1, 3, 6, 9]):
        super(GPD, self).__init__()
        self.expansion = expansion
        self.expand_conv = ConvBNReLU(in_channels, in_channels*expansion//2, ksize=1, pad=0, use_bn=False)
        self.reduce_conv = ConvBNReLU(in_channels*expansion//2, in_channels, ksize=1, pad=0, use_bn=False)
        #self.bn1 = nn.BatchNorm2d(in_channels*expansion//2)
        #self.bn2 = nn.BatchNorm2d(in_channels)
        self.end_conv = ConvBNReLU(in_channels, in_channels, use_relu=True, use_bn=False)
        self.se_block = SEBlock(in_channels*expansion//2)
        self.dilation_convs = nn.ModuleList()
        for i in dilation:
            self.dilation_convs.append(nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=i, dilation=i))
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
      
    def forward(self, x):
        y = self.expand_conv(x)
        y = torch.split(y, x.shape[1]//2, dim=1)
        res = []
        for idx, dilation_conv in enumerate(self.dilation_convs):
            res.append(dilation_conv(y[idx]))
        res = torch.cat(res, dim=1)
        #res = self.bn1(res)
        res = self.act1(res)
        res = self.se_block(res)
        res = self.reduce_conv(res)
        res = self.end_conv(res)
        return res


class FuseModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se1 = SEBlock(in_channels, in_channels)
        self.conv1 = ConvBNReLU(in_channels, in_channels, use_bn=False)
        self.se2 = SEBlock(in_channels * 3 // 2, in_channels * 3 // 2)
        self.reduce_conv = ConvBNReLU(in_channels * 3 // 2, in_channels, ksize=1, pad=0, use_bn=False)
        self.conv2 = ConvBNReLU(in_channels, in_channels, use_bn=False)

    def forward(self, low, high):
        x = self.se1(torch.cat([low, high], dim=1))
        x = self.conv1(x)
        x = self.se2(torch.cat([x, high], dim=1))
        x = self.reduce_conv(x)
        x = self.conv2(x)
        return x


class ImprovedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedDecoder, self).__init__()
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0, use_bn=False))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0, use_bn=False))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0, use_bn=False))

        self.fuse = nn.ModuleList()
        for i in range(len(in_channels)-1):
            self.fuse.append(FuseModule(out_channels[i]))
        self.fuse.append(ConvBNReLU(out_channels[-1], out_channels[-1], use_bn=False))

    def forward(self, features):
        stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](inner_lateral, inner_top_down) # low, high
            results.insert(0, stage_result)

        return results

