import torch
import torch.nn as nn
import torch.nn.functional as F

#import backbones
from model.vgg_backbone import VGGWrapper
from model.resnest_backbone import ResNestWrapper
#import segmentation head
from model.segmentation_head import SegmentationHead
#import prediction heads
from model.prediction_head_1D import PredictionHead1D
from model.prediction_head_2D import PredictionHead2D
from model.prediction_head_3D import PredictionHead3D


'''
Segression Block
Link: <Enter Arxiv Link upon publication>
Github: https://github.com/rajatmodi62/Segression
Input: [B,in_channels,H,W]
        Backbone: Input Backbone
                Args: VGG/ResNest
        segression_dimension: No of Parameters on which contours are drawn in prediction head.
                Args: 1,2,3
        epsilon: Perturbation in the model to prevent gaussians from exploding
Output: 2x[B,1,H,W], (Segmentation map & Gaussian Map)
Note: H/W should be a valid multiple of 32.
'''
class Segression(nn.Module):

    def __init__(self,\
                 in_channels=3,\
                 out_channels=32,\
                 backbone='VGG',\
                 segression_dimension= 3,\
                 center_line_segmentation_threshold= 0.7,\
                 gaussian_segmentation_threshold = 0.7,\
                 epsilon= 1e-7,\
                 pretrained=True):
        super(Segression, self).__init__()

        #store parameters.
        self.epsilon= epsilon

        ########################################################################
        #initialize backbone
        if backbone is 'VGG':
            self.backbone= VGGWrapper(in_channels=in_channels,\
                                      out_channels=out_channels,\
                                      pretrained= pretrained)
        elif backbone is "ResNest":
            self.backbone= ResNestWrapper(in_channels=in_channels,\
                                      out_channels=out_channels,\
                                      pretrained= pretrained)
        else:
            raise Exception("Backbone input not valid")

        ########################################################################
        #initialize segmentation_head
        #Note: in_channel=out_channel since it's plugged in front of backbone.
        self.segmentation_head= SegmentationHead(in_channels= out_channels)
        ########################################################################


        ########################################################################
        #initialize prediction heads
        #Note: in_channel=out_channel since it's plugged in front of backbone.
        if segression_dimension==1:
            self.prediction_head= PredictionHead1D(segmentation_map_threshold= center_line_segmentation_threshold,\
                                                   gaussian_segmentation_threshold= gaussian_segmentation_threshold,\
                                                   epsilon=epsilon)
        elif segression_dimension==2:
            self.prediction_head= PredictionHead2D(segmentation_map_threshold= center_line_segmentation_threshold,\
                                                   gaussian_segmentation_threshold= gaussian_segmentation_threshold,\
                                                   epsilon=epsilon)
        elif segression_dimension==3:
            self.prediction_head= PredictionHead3D(segmentation_map_threshold= center_line_segmentation_threshold,\
                                                   gaussian_segmentation_threshold= gaussian_segmentation_threshold,\
                                                   epsilon=epsilon)
        else:
            Exception("Dimensionality of gaussian should lie between 1 and 3")
        ########################################################################

        #initialize variance conv before prediction heads.
        self.variance_conv= nn.Conv2d(out_channels, segression_dimension, 1)

        print("Segression module initialized")

if __name__ == '__main__':
    segression= Segression()
