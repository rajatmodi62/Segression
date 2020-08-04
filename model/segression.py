import torch
import torch.nn as nn
import torch.nn.functional as F

#import backbones
from model.vgg_backbone import VGGWrapper
from model.resnest_backbone import ResNestWrapper
from model.DB.db_backbone import SegDetectorModel
#import segmentation head
from model.segmentation_head import SegmentationHead
#import prediction heads
from model.prediction_head_1D import PredictionHead1D
from model.prediction_head_2D import PredictionHead2D
from model.prediction_head_3D_vectorized import PredictionHead3D

#add edge detection
from model.edge_detection import EdgeDetection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                 n_classes=1,\
                 mode= 'train',\
                 center_line_segmentation_threshold= 0.7,\
                 gaussian_segmentation_threshold = 0.7,\
                 epsilon= 1e-7,\
                 pretrained=True,\
                 attention=False,\
                 attention_threshold= 2):
        super(Segression, self).__init__()

        #store parameters.
        self.epsilon= epsilon
        self.in_channels= in_channels
        self.out_channels= out_channels
        self.mode= mode
        self.attention= attention
        self.attention_threshold= attention_threshold
        self.n_classes = n_classes
        #initialize backbone
        if backbone =='VGG':
            self.backbone= VGGWrapper(in_channels=in_channels,\
                                      out_channels=out_channels,\
                                      pretrained= pretrained)
        elif backbone == "ResNest":
            self.backbone= ResNestWrapper(in_channels=in_channels,\
                                      out_channels=out_channels,\
                                      pretrained= pretrained)

        elif backbone== "DB":

            self.backbone=SegDetectorModel()
            #for training regarding ICDAR
            #checkpoint= 'snapshots/db_pretrained/ic15_resnet50'

            checkpoint= 'snapshots/db_pretrained/totaltext_resnet50'
            old_state_dict= torch.load(checkpoint,map_location=device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in old_state_dict.items():
                name= k[:6]+k[13:]
                new_state_dict[name] = v

            self.backbone.load_state_dict(new_state_dict,strict=True)
            print("loaded db backbone at:",checkpoint)
        else:
            raise Exception("Backbone input should be VGG/ResNest/DB")

        ########################################################################
        #initialize segmentation_head
        #Note: in_channel=out_channel since it's plugged in front of backbone.
        self.segmentation_head_1= SegmentationHead(in_channels= out_channels+2, n_classes= self.n_classes)
        ########################################################################


        ########################################################################
        #initialize prediction heads
        ########################################################################
        #Note: in_channel=out_channel since it's plugged in front of backbone.
        if segression_dimension==1:
            print("prediction head 1D")

            self.prediction_head= PredictionHead1D(segmentation_map_threshold= center_line_segmentation_threshold,\
                                                   gaussian_segmentation_threshold= gaussian_segmentation_threshold,\
                                                   epsilon=epsilon)
        elif segression_dimension==2:
            print("prediction head 2d")
            self.prediction_head= PredictionHead2D(segmentation_map_threshold= center_line_segmentation_threshold,\
                                                   gaussian_segmentation_threshold= gaussian_segmentation_threshold,\
                                                   epsilon=epsilon)
        elif segression_dimension==3:
            print("prediction head 3d")
            self.prediction_head= PredictionHead3D(segmentation_map_threshold= center_line_segmentation_threshold,\
                                                   gaussian_segmentation_threshold= gaussian_segmentation_threshold,\
                                                   epsilon=epsilon)
        else:
            Exception("Dimensionality of gaussian should lie between 1 and 3")
        ########################################################################

        #initialize variance conv before prediction heads.
        self.variance_conv= nn.Conv2d(out_channels, segression_dimension, 1)


        #######################################################################
        ########################################################################
        #self.edge_detection= EdgeDetection()

        #######################################################################
        print("Segression module initialized")

    '''
    Forward Pass for segression
    Input:
            X: image tensor [B,in_channel,H,W]
            Segmentation Map: [B,1,H,W] (non thresholded)
    Gaussians are drawn using segmentation map
    '''
    def forward(self,x,segmentation_map=None):
        ########################################################################
        #check input dimensions
        assert x.size()[1]==self.in_channels,\
            "input image channels does not match in_channels"
        assert x.size()[2]%32==0,\
            "Height dimension should be a multiple of 32"
        assert x.size()[3]%32==0,\
            "Width dimension should be a multiple of 32"
        ########################################################################

        #make a pass through backbone
        x,low_res_feature= self.backbone(x)
        backbone_feature= x
        #compute variance map
        variance= self.variance_conv(x)

        #attention based module
        ########################################################################
        if self.attention:
            variance_x= F.relu(variance[:,0,:,:])+1
            variance_y= F.relu(variance[ :,1,:,:])+1
            sum = variance_x+ variance_y
            # print('max value ', torch.max(sum))
            variance_attention= torch.gt(sum,self.attention_threshold)*1.0
            # print(torch.unique(variance_attention))
            #variance_attention = variance_attention.unsqueeze(1).repeat(1,x.shape[1],1,1)
            x= x*variance_attention.unsqueeze(1)
        x = torch.cat([F.relu(variance[:,0:2,...])+1, x], dim=1)

        ########################################################################

        #compute center line segmentation
        center_line_segmentation= self.segmentation_head_1(x,low_res_feature)



        #if mode is train, segmentation map used to create gaussians
        if self.mode== 'train':
            # if no external map given, use center_line_segmentation to compute gaussians.

            if segmentation_map is None:

                #call a code to perform skeletonization on center_line_segmentation (Future Edit)

                gaussian_segmentation= self.prediction_head(variance_map= variance,\
                                                            segmentation_map= center_line_segmentation)
                #skeletonization map
            else:
                #use externally provided map for training
                gaussian_segmentation= self.prediction_head(variance_map=variance,\
                                                            segmentation_map=segmentation_map)
            #print("backbone pass",x.size())

            #edge= self.edge_detection(gaussian_segmentation)
            #print("==========fwd pass done")
            meta = {
                'variance_attention':1
            }
            if self.attention:
                meta['variance_attention']= variance_attention

            return gaussian_segmentation,center_line_segmentation, variance,meta



        elif self.mode== 'test':
            #dont return gaussian map since it is explicitly drawn in the training code
            #print("performing testing")
            return center_line_segmentation,variance,backbone_feature

        else:
            raise Exception("Model should be operated in train/test mode only")
if __name__ == '__main__':
    print("hare krishna")
    model=Segression(center_line_segmentation_threshold=0.999,\
                    backbone="VGG",\
                    segression_dimension= 3,\
                    n_classes=1,\
                    attention=False,\
                    ).to(device)
    x= torch.randn(2,3,512,256).to(device)
    segmentation_map=torch.randn(1,1,128,64).to(device)
    gaussian_segmentation,center_line_segmentation, variance,meta= model(x,segmentation_map)
