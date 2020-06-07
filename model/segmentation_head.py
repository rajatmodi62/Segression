import torch
import torch.nn as nn
import torch.nn.functional as F

#To Do: Implement ASPP code later.
'''
SegmentationHead
Function: Performs segmentation on input feature map.

'''
class SegmentationHead(nn.Module):
    def __init__(self,in_channels=32):

        super(SegmentationHead, self).__init__()
        #convolution for segmentation.
        self.segmentation_conv= nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):

        x= F.sigmoid(self.segmentation_conv(x))
        return x

if __name__ == '__main__':
     print("main")
     seg= SegmentationHead()
     x= torch.randn(2,32,256,512)
     output=seg(x)
     #print("len of output",len(output))
     print("done",output.shape)
