import cv2
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util.augmentation import BaseTransform

def pil_load_img(path):
	image = Image.open(path).convert("RGB")
	image = np.array(image)
	return image

'''
Input: Dataset which is to be tested
'''

#give input of either scaling factors or scales
#scales: square 
#scaling_factor: rectangle 

def get_test_img_path(dataset):
	
	if dataset=='CTW1500':
		test_img_dir= 'data/ctw-1500/Images/Test'

	elif dataset=='MSRATD500':
		test_img_dir= 'data/msra-td500/Images/Test'

	elif dataset=='ICDAR2015':
		test_img_dir= 'data/icdar-2015/Images/Test'

	elif dataset=='TOTALTEXT':
		test_img_dir= 'data/total-text/Images/Test'

	else:
		raise Exception("Invalid Dataset given")

	#update the testing dir to the passed test_dir otherwise 

	test_img_path= [os.path.join(test_img_dir,path) \
						for path in os.listdir(test_img_dir)]
	
	test_img_path= sorted(test_img_path)

	return test_img_path

class TestDataLoader():

	def __init__(self,test_img_path):
		
		super().__init__()
		#get the list of image path
		self.means = (0.485, 0.456, 0.406)
		self.stds = (0.229, 0.224, 0.225)
		self.test_img_path= test_img_path
		#print("total images :",len(self.test_img_path))


	def __len__(self):
		return len(self.test_img_path)

	#copied from east
	
	def get_compatible_scaling_factor(self,h,w):
		#print("in compatible scalinig factor",h,w)
		resize_h = h
		resize_w = w
		resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
		resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
		#img= cv2.resize(img,(resize_w,resize_h), interpolation=cv2.INTER_NEAREST)
		# ratio_h = resize_h / h
		# ratio_w = resize_w / w
		# return ratio_h, ratio_w,resize_h,resize_w
		return resize_h,resize_w
	# scaling factors should be a list <1

	def getitem(self,\
					idx,\
					h_max=512,\
					w_max=512,\
					scaling_factors=[1],\
					hardcode=False,\
					hardcoded_scales=[]):

		image = pil_load_img(self.test_img_path[idx])
		img_path=self.test_img_path[idx]
		#print("in loader img path is ",img_path)
		img_id= img_path.split('/')[-1]
		h_orig, w_orig, _ = image.shape

		#find nearest multiple of 32 for h_max,w_max
		h_max,w_max= self.get_compatible_scaling_factor(h_max,w_max)

		#resize original image to h_max,w_max
		image= cv2.resize(image,(w_max,h_max), interpolation=cv2.INTER_LINEAR)

		#sort the input scaling factors 
		scaling_factors= sorted(scaling_factors)

		#scale images from h_max,w_max by scaling_factor.
		scaled_images= []
		scales=[]
		
		if hardcode==False:
			print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@FALSE HARDCODE')
			for scaling_factor in scaling_factors:
				
				h,w = self.get_compatible_scaling_factor( int(h_max*scaling_factor),\
													int(w_max*scaling_factor)
													)
				# scaled_image= cv2.resize(image,\
				# 					(w,h), \
				# 					interpolation=cv2.INTER_LINEAR)
							
				#intialize transform 
				transform = BaseTransform(
										size=(h,w), mean=self.means, std=self.stds
										)
				#print("applying transform")
				# print("type",type(transform(scaled_image)),len(transform(scaled_image)))
				scaled_images.append(transform(image)[0])
				scales.append((h,w))
		if hardcode==True:
			for scale in hardcoded_scales:
				
				# h,w = self.get_compatible_scaling_factor( int(h_max*scaling_factor),\
				# 									int(w_max*scaling_factor)
				# 									)
				# scaled_image= cv2.resize(image,\
				# 					(scale,scale), \
				# 					interpolation=cv2.INTER_LINEAR)
							
				#intialize transform 
				transform = BaseTransform(
										size=(scale,scale), mean=self.means, std=self.stds
										)
				#print("applying transform")
				# print("type",type(transform(scaled_image)),len(transform(scaled_image)))
				scaled_images.append(transform(image)[0])
				scales.append((scale,scale))
		#print(scales)

		meta = {
			'image_id': [img_id],
			'image_path': img_path,
			'image_shape_orig' : (h_orig,w_orig),
			'scales': scales

		}
		return scaled_images,meta

if __name__ == '__main__':

	print("main!!")
	
	test_img_path= get_test_img_path(dataset='TOTALTEXT')
	#initialize loader

	testset= TestDataLoader(test_img_path=test_img_path)

	#initialize scaling factors 
	scaling_factors= [0.25,0.50,0.75,1]
	
	#read testing images for H,W to be passed through 
	for i,img_path in enumerate(test_img_path):
		
		print("before img_path is",img_path)

		#read image to get h_orig,w_orig.
		image = pil_load_img(test_img_path[i])
		h_orig, w_orig, _ = image.shape

		#get the scaled images for testloader 
		# scaled_images,meta= testset.__getitem__(i,\
		# 										h_max=h_orig,\
		# 										w_max=w_orig,\
		# 										scaling_factors=scaling_factors)
		scaled_images,meta= testset.getitem(i,\
												h_max=512+2*128,\
												w_max=512+2*128,\
												scaling_factors=scaling_factors)
		input('halt')
	# for i in range(len(testset)):
	# 	testset.__getitem__(i,2)
	# 	print("i",i)


	########################
	
