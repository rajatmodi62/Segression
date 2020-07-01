import torch
import torch.nn as nn
import torch.nn.functional as F

#To Do: Implement ASPP code later.
'''
SegmentationHead
Function: Performs segmentation on input feature map.

'''

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class SegmentationHead(nn.Module):
    def __init__(self,\
            in_channels=32+2,\
             n_classes=1,\
             aspp_in_channels= 512):

        super(SegmentationHead, self).__init__()
        #convolution for segmentation.
        self.n_classes = n_classes
        self.conv1= nn.Conv2d(in_channels+(aspp_in_channels//2),in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2= nn.Conv2d(in_channels,n_classes, 1)
        self.aspp = ASPP(aspp_in_channels, depth=aspp_in_channels//2)

    def forward(self, x,low_res_feature):
        #print('helloxyz',x.shape)
        #print("rajat")
        size = x.shape[2:]
        #print("low res features size",low_res_feature.size())
        aspp_output = self.aspp(low_res_feature)
        aspp_output = F.upsample(aspp_output, size=size, mode='bilinear')
        #print("")
        x= torch.cat([x,aspp_output], dim=1)
        x= self.conv1(x)
        x= F.relu(self.bn1(x))
        x= self.conv2(x)
        #print("fata")
        if self.n_classes>1:
            x = F.softmax(x, dim=1)
        else:
            x = F.sigmoid(x)
        #x= F.sigmoid(self.conv1(x))
        return x

if __name__ == '__main__':
     print("main")
     seg= SegmentationHead()
     x= torch.randn(2,32,256,512)
     output=seg(x)
     #print("len of output",len(output))
     print("done",output.shape)
