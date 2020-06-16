import torch
import torch.nn as nn
import torch.nn.functional as F
#import encoder module
from model.DB.resnet import deformable_resnet18,deformable_resnet50
#import decoder module
from model.DB.seg_detector import SegDetector

#
# class BasicModel(nn.Module):
#     def __init__(self,in_channels=3,out_channels=32,pretrained=True):
#         nn.Module.__init__(self)
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.backbone= deformable_resnet50(pretrained=pretrained)
#         self.decoder= SegDetector(in_channels=[256, 512, 1024, 2048])
#
#     def forward(self, x, *args, **kwargs):
#         x=self.encoder(x)
#         x=self.decoder(x)
#         return x
#         #return self.decoder(self.encoder(data), *args, **kwargs)
#
# class SegDetectorModel(nn.Module):
#     def __init__(self, in_channels=3,out_channels=32,pretrained=True):
#         super(SegDetectorModel, self).__init__()
#         print('--------------->RAJAT,calling model')
#         self.model = BasicModel(in_channels=3,out_channels=32,pretrained=True)
#         # for loading models
#
#     def forward(self, x):
#         x = self.model(x)
#
#         print("rajat,x",x.size())
#         return x

#
# class DBWrapper(nn.Module):
#     def __init__(self,in_channels=3,out_channels=32,pretrained=True):
#         super(DBWrapper, self).__init__()
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.encoder= deformable_resnet50(pretrained=pretrained)
#         self.decoder= SegDetector(in_channels=[256, 512, 1024, 2048])
#



class BasicModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        #print("in basic model",args['backbone'])
        #print("in basic model",args['decoder'])
        # self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        # self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))
        self.backbone= deformable_resnet50(pretrained=True)
        self.decoder= SegDetector(in_channels=[256, 512, 1024, 2048])
    def forward(self, data):
        #print("@@@@@@@@@@@@@@@@@@rajat forward pass")
        return self.decoder(self.backbone(data))




class SegDetectorModel(nn.Module):
    def __init__(self):
        super(SegDetectorModel, self).__init__()

        self.model = BasicModel()

    def forward(self, batch, training=True):

        #print("--------------__> Rajat forward pass")
        pred = self.model(batch)

        return pred



if __name__ == '__main__':
        print("main")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        db=SegDetectorModel().to(device)
        checkpoint= 'snapshots/db_pretrained/ic15_resnet50'

        old_state_dict= torch.load(checkpoint,map_location=device)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        # create new OrderedDict that does not contain `module.`

        for k, v in old_state_dict.items():
            name= k[:6]+k[13:]
            new_state_dict[name] = v
        db.load_state_dict(new_state_dict,strict=True)



        x= torch.randn(2,3,256,512).cuda()
        out=db(x)
        print("out side",out.size())
