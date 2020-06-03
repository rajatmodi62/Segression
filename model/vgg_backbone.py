import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

'''
Making VGG Model Layers
'''
def make_layers(cfg, in_channels=3,batch_norm=False):
	layers = []
	#in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(3, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)

class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

'''
VGGEncoder
Function: feature extraction
Input: in_channels
Output: List of tensors for merging in the decoder
'''
class VGGEncoder(nn.Module):
	def __init__(self, in_channels=3,pretrained=True ):
		super(VGGEncoder, self).__init__()
		vgg16_bn = VGG(make_layers(cfg,in_channels, batch_norm=True))
		if pretrained :
			print("Loaded VGG16 Encoder's weights")
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features

	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]

'''
VGGDecoder
Function: Merge extracted features from Encoder
'''
class VGGDecoder(nn.Module):
	def __init__(self,out_channels=32):
		super(VGGDecoder, self).__init__()
		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, out_channels, 1)
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
VGGWrapper
Function: Wraps the encoder & decoder branches in a single module
Input: [B,in_channels,H,W]

Output: [B,out_channels,H,H]
'''
class VGGWrapper(nn.Module):
    def __init__(self,in_channels=3,out_channels=32,pretrained=True):
        super(VGGWrapper, self).__init__()

        #initialize encoder
        self.encoder= VGGEncoder(in_channels=in_channels,\
                                 pretrained=pretrained)
if __name__ == '__main__':
        print("main")
        vgg=VGGWrapper()
        print("done")
