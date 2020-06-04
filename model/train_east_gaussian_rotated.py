import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torchviz import make_dot
import matplotlib.pyplot as plt
import numpy as np
from model.resnest.resnet import ResNet, Bottleneck

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

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
def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
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


class extractor_vgg(nn.Module):
	def __init__(self, pretrained):
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		super(extractor_vgg, self).__init__()
		if pretrained:
			#print("rjat loading model")
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features

	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]

class merge_vgg(nn.Module):
	def __init__(self):
		super(merge_vgg, self).__init__()
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

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
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


class extractor_resnest(nn.Module):
	def __init__(self, pretrained):
		super(extractor_resnest, self).__init__()
		model = ResNet(Bottleneck, [3, 4, 6, 3],
	                   radix=2, groups=1, bottleneck_width=64,
	                   deep_stem=True, stem_width=32, avg_down=True,
	                   avd=True, avd_first=False)
		if pretrained:
			model.load_state_dict(torch.hub.load_state_dict_from_url(
	            resnest_model_urls['resnest50'], progress=True))
		self.features = model

	def forward(self, x):
		feature_maps = self.features(x)
		return feature_maps

class merge_resnest(nn.Module):
	def __init__(self):
		super(merge_resnest, self).__init__()
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

		self.conv5 = nn.Conv2d(320, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
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



'''gaussian projection'''
'''project feature maps to two dimensions'''
'''treat each element as a mean & variance '''
''' '''
class gaussian_projection(nn.Module):
	def __init__(self,in_channel,segmentation_threshold=0.7):
		super(gaussian_projection,self).__init__()
		self.segmentation_threshold=segmentation_threshold
		self.sigmoid= nn.Sigmoid()
		#adding the dynamic thresholding layer
		self.dynamic_threshold= nn.Conv2d(in_channel,1,1)
		self.dynamic_activation= nn.Sigmoid()

	def create_gaussian_tensor(self,variance_x,variance_y,theta,x,y,height,width,current_threshold):

		#rotated gaussians
		#define the angular values
		sin = torch.sin(theta)
		cos = torch.cos(theta)
		var_x = torch.pow(variance_x,2)+1e-7
		var_y = torch.pow(variance_y,2)+1e-7
		#print("==============UNIVERSITY OF SURREY",var_x,var_y)
		a= (cos*cos)/(2*var_x) + (sin*sin)/(2*var_y)
		b= (-2*sin*cos)/(4*var_x) + (2*sin*cos)/(4*var_y)
		c= (sin*sin)/(2*var_x) + (cos*cos)/(2*var_y)
		#
		i= torch.linspace(0,height-1,height)#.cuda()
		i=i.unsqueeze(1)
		i= i.repeat(1,width)
		j=i.T

		#the x & y around which gaussian is centered
		A= torch.pow(i-x,2)
		#B=1
		B= 2*(i-x)*(j-y)
		C= torch.pow(j-y,2)
		gaussian_tensor= torch.exp(-(a*A+b*B+c*C)+1e-7)#.cuda()
		del a,b,c,A,B,C,i,j
		return gaussian_tensor

	def forward(self,x,variance,center_map):


		#calculate the threshold map
		threshold_map= self.dynamic_activation(self.dynamic_threshold(x))

		batch_size,_,_,_=center_map.size()
		batch_outputs=[]
		flag=False

		#print("entered forward",batch_size)
		#loop through the batch
		for i in range(batch_size):
			batch_center_map=center_map[i,0,:,:]
			#print("seg threshold",self.segmentation_threshold)
			batch_center_map=(batch_center_map>self.segmentation_threshold).float()#.cuda()
			#print(batch_center_map)
			batch_variance_x= variance[i,0,:,:]
			batch_threshold= threshold_map[i,0,:,:]
			batch_variance_x=(batch_variance_x*batch_center_map)
			batch_variance_y= variance[i,1,:,:]
			batch_variance_y= (batch_variance_y*batch_center_map)
			# print("==================+>BATCH ",batch_variance_x[x_coordinate][y_coordinate],batch_variance_y[x_coordinate][y_coordinate])
			h,w=batch_variance_x.size()
			batch_gaussian_tensors=[]
			batch_theta_map= 3.14*F.sigmoid(variance[i,2,:,:])

			#it does not matter whose non zero values are taken, since both may be  non zero
			non_zero_tensors=torch.nonzero(batch_variance_x+ batch_variance_y)

			no_of_nonzero_variances= non_zero_tensors.size()[0]
			#print(no_of_nonzero_variances)
			#print("total non zero variances are",no_of_nonzero_variances)

			if no_of_nonzero_variances==0:
				flag=True
				return batch_outputs, flag

			for j in range(no_of_nonzero_variances):
				#print(j)
				x_coordinate=non_zero_tensors[j,0].item()
				y_coordinate= non_zero_tensors[j,1].item()
				current_threshold= batch_threshold[x_coordinate][y_coordinate]
				batch_gaussian_tensors.append(self.create_gaussian_tensor(batch_variance_x[x_coordinate][y_coordinate],batch_variance_y[x_coordinate][y_coordinate],batch_theta_map[x_coordinate][y_coordinate],x_coordinate,y_coordinate,h,w,current_threshold))
				#print("created tensor",j)
			batch_gaussians_tensors=torch.stack(batch_gaussian_tensors,0)
			batch_gaussians_tensors=torch.max(batch_gaussians_tensors,0).values
			#thresholding debugging
			#print("type",type(batch_gaussians_tensors))
			#batch_gaussians_tensors=(batch_gaussians_tensors>=0.7).float().cuda()*batch_gaussians_tensors
			batch_gaussians_tensors=(batch_gaussians_tensors>=0.7).float()*batch_gaussians_tensors
            #implementing the dynamic threshold for the segmentation
			#batch_gaussians_tensors=(batch_gaussians_tensors>=threshold_map).float().cuda()*batch_gaussians_tensors
			#batch_gaussians_tensors= F.relu(batch_gaussians_tensors)
			batch_gaussians_tensors=batch_gaussians_tensors.unsqueeze(0)
			batch_outputs.append(batch_gaussians_tensors)
		#print("returning one forward pass")
		batch_outputs=torch.stack(batch_outputs,0)
		#print("batch outputs",batch_outputs)
		#variance map needs to be thresholded multiple times
		return batch_outputs, flag

class prediction_head(nn.Module):
	def __init__(self,segmentation_threshold=0.7,segmentation_attention=False):

		super(prediction_head, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.segmentation_threshold= segmentation_threshold
		self.gaussian_projection1 = gaussian_projection(32,segmentation_threshold)
		self.sigmoid2= nn.Sigmoid()
		#external ground truth map for switching training dynamically.
		self.external_label_map= None
		self.segmentation_attention= segmentation_attention

		#intialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		#variance convolution
		self.variance3D= nn.Conv2d(32,3,1)

	def forward(self, x):
		#calculate variance
		variance= self.variance3D(x)
		#print("variance is",variance.size())
		#threshold the variance map
		variance_thresholds= [10,7,5]

		#changed variance map has to pass through segmentation
		changed_variance_map=  []
		# print("unique variance_x",torch.min(variance_x),torch.max(variance_x))
		# print("unique variance_y",torch.min(variance_y),torch.max(variance_y))

		for threshold in variance_thresholds:
			variance_x= variance[:,0,:,:]
			variance_y= variance[:,1,:,:]
			# print("unique variance_x",torch.min(variance_x),torch.max(variance_x))
			# print("unique variance_y",torch.min(variance_y),torch.max(variance_y))
			variance_x= (variance_x>threshold)*1.0
			variance_y= (variance_y>threshold)*1.0
			changed_variance_map.append(variance_x + variance_y)
			#changed_variance_map.append((variance*-1>threshold)*1.0)



		#changed_variance_map = torch.mean(torch.stack(changed_variance_map),axis=0)
		#if self.segmentation_attention:
			#score= x*(changed_variance_map.unsqueeze(1).repeat(1,x.shape[1],1,1))
		score= self.conv1(x)
		score= self.sigmoid1(score)

		#calculate gaussians ->contour map from original variance map
		if self.external_label_map is not None:
			# print("external map is not none")
			contour_map, flag = self.gaussian_projection1(x,variance,self.external_label_map)
		else:
			#print("external map is none")
			contour_map, flag= self.gaussian_projection1(x,variance,score)

			#contour_map= self.sigmoid2(outx)
		self.external_label_map= None
		return score,contour_map, flag,variance

	def switch_gaussian_label_map(self,label_map= None):
		self.external_label_map = label_map

class Segression(nn.Module):
	def __init__(self,segmentation_threshold=0.7,pretrained=True,segmentation_attention=False,backbone='vgg'):
		super(Segression, self).__init__()
		if backbone=='vgg':
			self.extractor = extractor_vgg(pretrained)
			self.merge     = merge_vgg()
		elif backbone=='resnest':
			self.extractor = extractor_resnest(pretrained)
			self.merge     = merge_resnest()
		self.segmentation_threshold=segmentation_threshold
		self.segmentation_attention= segmentation_attention
		self.prediction_head    = prediction_head(segmentation_threshold,self.segmentation_attention)


	def forward(self, x):
		x= self.extractor(x)
		x= self.merge(x)
		x=self.prediction_head(x)
		return x
		# return self.output(self.merge(self.extractor(x)))

	def switch_gaussian_label_map(self,label_map= None):
		self.prediction_head.switch_gaussian_label_map(label_map)

if __name__ == '__main__':

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#device= torch.device("cpu")

	#m = Segression(segmentation_threshold=0.999, backbone='resnest').cuda()
	m = Segression(segmentation_threshold=0.999, backbone='resnest').to(device)

	# x = torch.randn(1, 3, 512, 512).to(device)
	# score,contour_map,flag,variance= m(x)
	# print("done",score.size(),contour_map.size(),variance.size())



	#print(make_dot(m(x)).source)
	#make_dot(m(x)).render('test.gv', view=True)

	# x=torch.randn(8,32,128,128)
	# centre_map=torch.randn(8,1,128,128)
	# m=gaussian_projection(32)
	# output=m(x,centre_map)
	# print(output.size())
