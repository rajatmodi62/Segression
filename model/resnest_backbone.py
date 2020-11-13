import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnest.resnet import ResNet, Bottleneck

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

'''
ResnestEncoder
Function: feature extraction
Input: in_channels
Output: List of tensors for merging in the decoder
'''

#handle in_channels later on.
class ResNestEncoder(nn.Module):
	def __init__(self, in_channels=3,pretrained=True):
		super(ResNestEncoder, self).__init__()
		self.in_channels= in_channels
		model = ResNet(Bottleneck, [3, 4, 6, 3],
	                   radix=2, groups=1, bottleneck_width=64,
	                   deep_stem=True, stem_width=32, avg_down=True,
	                   avd=True, avd_first=False)
		if pretrained:
			print("loading pretrained weights")
			model.load_state_dict(torch.hub.load_state_dict_from_url(
	            resnest_model_urls['resnest50'], progress=True))
			print("pretrained weights loaded")

		self.features = model

	def forward(self, x):
		feature_maps = self.features(x)
		return feature_maps

'''
ResNestDecoder
Function: Merge extracted features from Encoder
'''

class ResNestDecoder(nn.Module):
	def __init__(self,out_channels=32):
		super(ResNestDecoder, self).__init__()
		self.conv1 = nn.Conv2d(3072, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(640, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(320, out_channels, 1)
		self.bn5 = nn.BatchNorm2d(out_channels)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(out_channels)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(out_channels)
		self.relu7 = nn.ReLU()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[2]), 1)
		y = self.relu1(self.bn1(self.conv1(y)))
		y = self.relu2(self.bn2(self.conv2(y)))

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))
		y = self.relu4(self.bn4(self.conv4(y)))

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))
		y = self.relu6(self.bn6(self.conv6(y)))

		y = self.relu7(self.bn7(self.conv7(y)))
		return y

'''
ResNestWrapper
Function: Wraps the encoder & decoder branches in a single module
Input: [B,in_channels,H,W]

Output: [B,out_channels,H,H]
'''
class ResNestWrapper(nn.Module):
    def __init__(self,in_channels=3,out_channels=32,pretrained=True):
        super(ResNestWrapper, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.encoder= ResNestEncoder(in_channels=self.in_channels,\
                                 pretrained=pretrained)
        self.decoder= ResNestDecoder(out_channels=self.out_channels)

    def forward(self, x):
         # assert x.size()[1]==self.in_channels,\
         #        "Initialized in_channel & Input Tensor's in_channel must be same"
         x= self.encoder(x)
         x= self.decoder(x)

         return x
if __name__ == '__main__':
     print("main")
     resnest=ResNestWrapper(pretrained=True)
     x= torch.randn(2,3,256,512)
     output=resnest(x)
     #print("len of output",len(output))
     print("done",output.shape)
