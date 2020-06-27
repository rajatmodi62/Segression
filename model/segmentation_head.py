import torch
import torch.nn as nn
import torch.nn.functional as F

#To Do: Implement ASPP code later.
'''
SegmentationHead
Function: Performs segmentation on input feature map.

'''
class SegmentationHead(nn.Module):
    def __init__(self,in_channels=32+2, n_classes=1):

        super(SegmentationHead, self).__init__()
        #convolution for segmentation.
        self.n_classes = n_classes
        self.conv1= nn.Conv2d(in_channels,in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2= nn.Conv2d(in_channels,n_classes, 1)

    def forward(self, x):
        #print('helloxyz',x.shape)
        x= self.conv1(x)
        x= F.relu(self.bn1(x))
        x= self.conv2(x)
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
