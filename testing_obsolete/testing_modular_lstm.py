import os
import time
import cv2
import numpy as np
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import statistics
import matplotlib.pyplot as plt
from skimage import measure
from packaging import version
from testing_obsolete.testloader_square import TestDataLoader
from testing_obsolete.create_eval_outputs import EvalOutputs
from testing_obsolete.find_longest_path import findLongestPath
from model.segression import Segression
from util.augmentation import BaseTransform
from shapely.geometry import Polygon
from skimage.morphology import medial_axis
from skeletonize import skeletonize_image
import shutil as sh
from pathlib import Path
import math
import torch.nn.functional as F
import argparse
from model.lstm import BreakLine_v2
from fil_finder import FilFinder2D
import astropy.units as u 


###############################################################################
parser = argparse.ArgumentParser(description="Welcome to the Segression Testing Module!!!")

parser.add_argument("--dataset", type=str, default="TOTALTEXT",
					help="Dataset on which testing is to be done, (TOTALTEXT,CTW1500,MSRATD500,ICDAR2015)")
parser.add_argument("--test-dir", type=str, default="",
					help="Directory upto test folder for the dataset")
			
parser.add_argument("--snapshot-dir", type=str, default="snapshots/SynthText_3d_rotated_gaussian_without_attention_200000.pth",
					help="Path to the Segression snapshot to be used for testing")
parser.add_argument("--lstm-snapshot-dir", type=str, default="snapshots/LSTM_batch_size_8lr_0.0001n_steps_100000dataset_TOTALTEXTbackbone_VGG/TOTALTEXT_LSTM_checkpoint22000.pth",
					help="Path to the LSTM Snapshot used for testing")
parser.add_argument("--segmentation_threshold", type=float, default=0.4,
					help="Thresholding parameter for predicted center line mask, range (0,1)")
parser.add_argument("--gaussian_threshold", type=float, default=0.6,
					help="Thresholding parameter for predicted gaussian map, range (0,1)")
parser.add_argument("--lstm_threshold", type=float, default=0.8,
						help="Enter lstm threshold")
parser.add_argument("--backbone", type=str, default="VGG",
					help="Enter the Backbone of the model, (VGG,RESNEST,DB)")
parser.add_argument("--out-channels", type=int, default=32,
						help="Save summaries and checkpoint every often.")
parser.add_argument("--n-classes", type=int, default=1,
						help="number of classes in segmentation head.")

parser.add_argument("--mode", type=str, default="",
					help="Enter training with/without lstm")
args = parser.parse_args()

#print("All requisite testing modules loaded")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#free the gpus
os.system("nvidia-smi | grep 'python' | awk '{ #print $3 }' | xargs -n1 kill -9")

#enter scales in a sorted increasing order
scales= [512,512+128,512+2*128,1024, 1024+256, 1024+512]
scaling_factor= np.max(np.asarray(scales))/scales

BATCH_SIZE = 1
INPUT_SIZE = 256

#define absolute path
local_data_path = Path('.').absolute()
local_data_path.mkdir(exist_ok=True)

''' this should be handled in deteval code'''
# if os.path.exists('./output.txt'):
#     os.remove('./output.txt')
#delete the result folder if exists

paths= [str(local_data_path/args.dataset/'results'/'predictions'),\
		str(local_data_path/args.dataset/'results'/'variance_maps'),\
		str(local_data_path/args.dataset/'results'/'filteredcontours'),\
		str(local_data_path/args.dataset/'results'/'center_line'),\
		str(local_data_path/args.dataset/'results'/'predictions_with_area'),\
		]

for path in paths:
	if os.path.isdir(path):
		sh.rmtree(path)


################################################################################


def area(x, y):
	polygon = Polygon(np.stack([x, y], axis=1))
	return float(polygon.area)

def create_gaussian_array(variance_x,variance_y,theta,x,y,height=INPUT_SIZE//4,width=INPUT_SIZE//4):
	sin = math.sin(theta)
	cos = math.cos(theta)
	var_x = math.pow(variance_x,2)
	var_y = math.pow(variance_y,2)
	a= (cos*cos)/(2*var_x) + (sin*sin)/(2*var_y)
	b= (-2*sin*cos)/(4*var_x) + (2*sin*cos)/(4*var_y)
	c= (sin*sin)/(2*var_x) + (cos*cos)/(2*var_y)
	i= np.array(np.linspace(0,height-1,height))
	i= np.expand_dims(i,axis=1)
	i=np.repeat(i,height,axis=1)
	j=i.T
	#the x & y around which gaussian is centered
	A= np.power(i-x,2)
	# B=1
	B= 2*(i-x)*(j-y)
	C= np.power(j-y,2)
	gaussian_array= np.exp(-(a*A+b*B+c*C))
	return gaussian_array

def variance_filter(variance_map, max_value_index, mask, pred_center_line, mode='average'):

	filtered_variance_map = torch.zeros(variance_map[0].shape).float()
	variance_map = torch.cat(variance_map, dim=0) # scalesx3xhxw
	pred_center_line = torch.cat(pred_center_line, dim=0)
	pred_center_line =(pred_center_line.squeeze()>args.segmentation_threshold)*1.0

	pred_center_line = pred_center_line.unsqueeze(1).repeat(1,3,1,1)
	add_center_line = torch.sum(pred_center_line,dim=0)
	mask = mask.squeeze()
	non_zero_index = mask.nonzero()
	scaling_factor= np.max(np.asarray(scales))/scales
	#print('scaling factor', scaling_factor)
	if mode=='average':
		scaling_factor = torch.from_numpy(np.asarray(scaling_factor)).cuda()
		scaling_factor = scaling_factor.reshape(-1,1,1,1).repeat(1,2,variance_map.shape[-2], variance_map.shape[-1])

	for index in non_zero_index:
		if mode=='average':
			slice_value = variance_map[:,:,index[0],index[1]]*pred_center_line[:,:,index[0],index[1]]
			slice_value[:,0:2]=slice_value[:,0:2]*scaling_factor[:,:,index[0],index[1]]
			slice_value=torch.sum(slice_value,dim=0)/add_center_line[:,index[0],index[1]]
			filtered_variance_map[0,:,index[0],index[1]]= slice_value#variance_map[mask[index[0],index[1]],:,index[0],index[1]]
		else:
			max_value= max_value_index[index[0],index[1]]
			slice_value = variance_map[max_value,:,index[0],index[1]]
			filtered_variance_map[0,:,index[0],index[1]]= slice_value#variance_map[mask[index[0],index[1]],:,index[0],index[1]]
			filtered_variance_map[0,0,index[0],index[1]]=filtered_variance_map[0,0,index[0],index[1]]*scaling_factor[max_value_index[index[0],index[1]]]
			filtered_variance_map[0,1,index[0],index[1]]=filtered_variance_map[0,1,index[0],index[1]]*scaling_factor[max_value_index[index[0],index[1]]]
	# filtered_variance_map = variance_map[max_value_index,:,:,:] # 3xhxw
	filtered_variance_map=filtered_variance_map.detach().cpu().numpy()
	# plt.imshow(filtered_variance_map[0][0])
	# plt.show()
	return filtered_variance_map

def voting(center_line_list):

	#get the minimum no of votes needed
	min_votes_needed= len(scales)//2 +  1
	
	#perform the sum in the list
	mask = torch.zeros(center_line_list[0].shape).to(device)

	for scaled_center_line_image in center_line_list:
		temp = (scaled_center_line_image.squeeze()>args.segmentation_threshold)*1.0
		mask+= temp

	#perform voting
	mask= (mask>= min_votes_needed)*1.0
	#.astype('uint8')

	center_line_list = torch.cat(center_line_list,dim=0) # scalesxhxw
	max_value_index = torch.argmax(center_line_list,dim=0)# hxw

	# plt.imshow(mask.cpu().numpy()[0])
	# plt.show()
	return mask, max_value_index

from numba import jit, prange

# @autojit
def construct_input_parallel( coordinate_sequence, feature_map):
	instance_length = coordinate_sequence.shape[0]
	input_lstm = torch.zeros(instance_length, 1,args.out_channels)
	##print("total instance length is",instance_length)
	##print("feature map shape",feature_map.shape)
	for len in prange(instance_length):
		coordinate = coordinate_sequence[len,:]   # x,y
		##print("coordinate being extracted is",coordinate)
		input_lstm [len,0,:] = feature_map[0,:,coordinate[0], coordinate[1]].squeeze()
	return  input_lstm,instance_length


def construct_input( coordinate_sequence, feature_map):
	instance_length = coordinate_sequence.shape[0]
	input_lstm = torch.zeros(instance_length, 1, args.out_channels)
	for len in range(instance_length):
		coordinate = coordinate_sequence[len,:]   # x,y
		input_lstm [len,0,:] = feature_map[0,:,coordinate[0], coordinate[1]].squeeze()
	return  input_lstm,instance_length


def load_model():
	#print("channels is",args.out_channels)
	model= Segression(center_line_segmentation_threshold=args.segmentation_threshold,\
						backbone=args.backbone,\
						segression_dimension= 3,\
						out_channels= args.out_channels,\
						n_classes=args.n_classes,\
						mode='test').to(device)
	
	#load checkpoint
	#print("trying to load snapshot: ", args.snapshot_dir)
	model.load_state_dict(torch.load(args.snapshot_dir,map_location=device),strict=True)

	#print("snapshot loaded!!!!")
	model.eval()
	if args.mode=='with_lstm':
		hidden_size=64
		lstm_model = BreakLine_v2( hidden_size=hidden_size,output_size=1,feature_feature=32).cuda()
		#print("trying to load lstm_model")
		lstm_model.load_state_dict(torch.load(args.lstm_snapshot_dir,map_location=device),strict=True)
		#print("lstm model loaded!!!")
		lstm_model.eval()
		return model,lstm_model

	return model 

def model_inference(model,image,max_scale):
	#forward pass
	with torch.no_grad():
		score_map, variance_map,backbone_feature= model(image)

		#score map is center line map
		if args.n_classes>1:
			conceal = F.sigmoid(score_map[:,2,:,:])
			score_map =  F.softmax(score_map[:,0:2,...],dim=1)
			score_map = score_map[:,0,:,:]*conceal
			#print('max value ', torch.max(score_map))
			#plt.imshow((score_map+conceal).squeeze().cpu().numpy()*200)
			#plt.show()
			score_map= score_map.unsqueeze(1)

		score_map=F.upsample(score_map, size=[max_scale,max_scale], mode='nearest')
		score_map= score_map.squeeze(0)
		score_map= score_map.squeeze(0)

		#variance map contains the variances of the gaussians
		variance_map = F.upsample(variance_map, size=[max_scale,max_scale], mode='nearest')
		variance_map= variance_map.squeeze(0)
		variance_map=variance_map.squeeze(0)

		theta_map=  3.14*F.sigmoid(variance_map[2,:,:])
		theta_map= theta_map.detach().cpu().numpy()
		return score_map.unsqueeze(0), variance_map.unsqueeze(0), theta_map, backbone_feature

def multiscale_aggregation(pred_center_line, pred_variance_map):
	#perform voting here
	score_map_maximum_scale= pred_center_line[-1]
	score_map, max_value_index = voting(pred_center_line)

	variance_map = variance_filter(pred_variance_map, max_value_index, score_map,pred_center_line)
	score_map = score_map.detach().cpu().numpy()
	score_map= (score_map>0).astype('uint8')

	#print("variance map shape", variance_map[0].shape,type(variance_map))
	return score_map, variance_map

def smoothen_center_line(score_map):	
	skeletons= []
	#for score_map in score_maps:
	skeleton= score_map
	fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
	fil.preprocess_image(flatten_percent=85)
	fil.create_mask(border_masking=True, verbose=False,
	use_existing_mask=True)
	fil.medskel(verbose=False)
	print("u.pix",)
	fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
	return fil.skeleton

	
	# skeletons.append(fil.skeleton)
	# # Show the longest path
	# # plt.subplot(1,2,1)
	# # plt.imshow(score_map)
	# # plt.subplot(1,2,2)
	# # plt.imshow(fil.skeleton, cmap='gray')
	# # plt.axis('off')
	# # plt.show()


def extract_center_line(score_map, smooth=True):
	#extract contours from it.
	blobs_labels = measure.label(score_map, background=0)
	ids = np.unique(blobs_labels)

	#iterate through score maps and extract a contour
	component_score_maps=[]
	component_center_line= []
	for component_no in range(len(ids)):
		if ids[component_no]==0:
			continue
		current_score_map= (blobs_labels==ids[component_no]).astype('uint8')*255
		current_score_map=current_score_map.astype('uint8')[0]
		#print(np.unique(current_score_map),"current score map",current_score_map.shape,current_score_map.dtype)
		contours,hierarchy=cv2.findContours(current_score_map, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

		#extract the single contour which is there
		if len(contours)==0:
			print("skeleton is empyt")
			continue
		contours= contours[0].squeeze(1)

		#scale up the points
		scaling=[scaling_factor_x,scaling_factor_y]
		scaling=np.array(scaling).T
		contours=contours*scaling

		#skip if a component has less than 2 points
		if contours.shape[0]<=2:
			continue

		#skeletonize the current score map
		current_score_map= skeletonize_image(current_score_map).astype('uint8')
		#print("Score map before shape",current_score_map.shape)
		if smooth:
			#print("code is in smooth")
			#print("score map after shape",current_score_map.shape)
			current_score_map =smoothen_center_line(current_score_map)
		#showing score map 
		# print("Score map")
		# plt.imshow(current_score_map)
		# plt.show()

		#current_score_map = medial_axis(current_score_map).astype('uint8')

		#print("current_score_map",current_score_map.shape)
		contours,hierarchy=cv2.findContours(current_score_map*255, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

		# #print(type(contours),contours[0].shape)
		if len(contours)==0:
			print("skeleton is empyt")
			continue
		contours= contours[0].squeeze(1)

		modified_score_map= current_score_map.copy()
		modified_score_map=cv2.resize(modified_score_map,((512+2*128)//4,(512+2*128)//4), interpolation=cv2.INTER_NEAREST)
		modified_score_map= (modified_score_map>0)*1.0
		component_center_line.append(contours)
		component_score_maps.append(current_score_map) 
	return component_score_maps, component_center_line
	
def construct_contours(variance_map, component_score_maps, theta_map):
	sample_contours=[]
	center_line_list=[]
	#print("len of list",len(sample_contours))
	#print("variance map shape",type(variance_map))
	variance_map= variance_map[0]
	variance_map_x= variance_map[0]
	variance_map_y= variance_map[1]
	# plt.subplot(1,2,1)
	# plt.imshow(variance_map_x)
	# plt.subplot(1,2,2)
	# plt.imshow(variance_map_y)
	# plt.show()
	#######################################
	#print("center lines are input",len(component_score_maps))
	for i,center_line_mask in enumerate(component_score_maps):

		# print("showing ceter line mask")
		# plt.imshow(center_line_mask)
		# plt.show()
		#print("before unique elements in center",np.unique(center_line_mask))
		#make the center_line_mask lie between 0 and 1 
		center_line_mask= (center_line_mask>0)*1.0
		#print("after unique elements in center",np.unique(center_line_mask))
		component_variance_map= center_line_mask*(variance_map_x+ variance_map_y)
		##print("full variance map")
		component_variance_map_x= center_line_mask*variance_map_x
		component_variance_map_y= center_line_mask*variance_map_y
		component_theta_map= center_line_mask*theta_map
		non_zero_arrays=np.nonzero(component_variance_map)
		n_gaussians= non_zero_arrays[0].shape[0]
		#print("i",i)
		component_gaussians=[]
		#max_scale= scales[-1]//4
		##print("n_gaussians",n_gaussians,"for image",image_id)
		if n_gaussians:
			##print("entering")
			for j in range(n_gaussians):
				x_coordinate=non_zero_arrays[0][j]
				y_coordinate= non_zero_arrays[1][j]
				gaussian=create_gaussian_array(component_variance_map_x[x_coordinate][y_coordinate],component_variance_map_y[x_coordinate][y_coordinate],component_theta_map[x_coordinate][y_coordinate],x_coordinate,y_coordinate,height=scales[-1]//4,width=scales[-1]//4)
				component_gaussians.append(gaussian)
				# if j%100 is 0:
					#print(j)
			component_gaussians= np.stack(component_gaussians,0)
			component_gaussians= np.max(component_gaussians,0)
			# plt.imshow(component_gaussians)
			# plt.show()
			component_gaussians= (component_gaussians>args.gaussian_threshold).astype(int)
			component_gaussians=component_gaussians.astype('uint8')

			contours,hierarchy=cv2.findContours(component_gaussians, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
			cv2.drawContours(visual_image, contours, 0, (255, 255, 255), 1)
			# print("visual image")
			# plt.imshow(visual_image)
			# plt.show()
			filtered_contour_list_by_points=[contours[k].squeeze(1) for k in range(len(contours)) if len(contours[k])>=25]
			filtered_contour_list=[k for k in filtered_contour_list_by_points if area(k[:,0].tolist(),k[:,1].tolist()) ]
			if len(filtered_contour_list)==0:
				sample_contours.append('')
				center_line_list.append('')
			else:
				for filtered_list in filtered_contour_list:
					sample_contours.append(filtered_list)
					center_line_list.append(component_center_line[i])

		else:
			sample_contours+=['']
			center_line_list+=['']
	return sample_contours, center_line_list

def scaling_to_original_scale(scaling_factor_x, scaling_factor_y,\
	filtered_contour_list,filtered_center_line_list):
	for j in range(len(filtered_contour_list)):

		scaling=[scaling_factor_x,scaling_factor_y]
		scaling=np.array(scaling).T
		##print("dtype",filtered_contour_list[0].dtype,type(filtered_contour_list[j]))
		if filtered_contour_list[j]=='':
			continue
		filtered_contour_list[j]=filtered_contour_list[j]*scaling
		filtered_contour_list[j] = filtered_contour_list[j].astype(int)
		
		filtered_center_line_list[j]=filtered_center_line_list[j]*scaling
		filtered_center_line_list[j] = filtered_center_line_list[j].astype(int)
	return filtered_contour_list, filtered_center_line_list


def draw_contour(contour,max_scale):
	canvas= np.zeros((max_scale,max_scale,3),dtype='uint8')
	cv2.drawContours(canvas, [contour], 0, (255,255, 255), 1)

	# plt.imshow(canvas)
	# plt.show()
	return canvas[:,:,0]

def refine_center_line(backbone_feature,center_lines,lstm_model, max_scale):
	##print('================== center line refined')
	#print(backbone_feature.shape)
	set_of_contours=[]
	for contours in center_lines:

		sequential_feature, length = construct_input_parallel( contours, backbone_feature)
		hidden = lstm_model.initHidden(1)
		output = torch.ones(1, 1, device=device)
		stacked_output=[]
		set_of_points=[]
		flag=True
		for index in range(length):
			output, hidden = lstm_model(sequential_feature[index,...].cuda(), hidden.cuda(), output.cuda())    
			##print(output)    
			output = (output>args.lstm_threshold)*1.0
			# #print(output)
			if int(output)==1:
				##print("output appended")
				stacked_output.append(output)
				set_of_points.append(contours[index,:])
			else:
				flag=False
				if len(set_of_points)>0:
					set_of_points = np.asarray(set_of_points)
					set_of_contours.append(set_of_points)
					set_of_points=[]
		if flag:
			set_of_points = np.asarray(set_of_points)
			# #print(set_of_points)
			set_of_contours.append(set_of_points)
	#print("=======================set of contours")
	#print("len of set of contours",len(set_of_contours))
	new_score_map=[]
	for contour in set_of_contours:
		#print('helllllllllllllooooooooooooo')
		new_score_map.append(draw_contour(contour,max_scale))
	return new_score_map, set_of_contours 
	
	
def check_continuity(component_score_map,component_center_line):
	#print("in continutity",type(component_score_map),component_score_map.shape)
	h,w= component_score_map.shape
	canvas= np.zeros((h,w,3))
	for step in range(component_center_line.shape[0]-2):
		#cv2.line(canvas,[component_center_line[step:step+2]],(0, 255, 0), thickness=2)
		##print("rajat",component_center_line[step][0],component_center_line[step][1]))
		cv2.line(canvas, (component_center_line[step][0],component_center_line[step][1])\
				, (component_center_line[step+1][0],component_center_line[step+1][1]), (0, 255, 0), thickness=2)	
		cv2.imshow('image',canvas)
		cv2.waitKey(500)
		cv2.destroyAllWindows()


def smoothen_center_line(score_map):	
	
	skeleton= score_map
	fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
	fil.preprocess_image(flatten_percent=85)
	fil.create_mask(border_masking=True, verbose=False,
	use_existing_mask=True)
	fil.medskel(verbose=False)
	#print("u.pix",)
	fil.analyze_skeletons(branch_thresh=10* u.pix, skel_thresh=1 * u.pix, prune_criteria='length')
		
	# Show the longest path
	# plt.subplot(1,2,1)
	# plt.imshow(score_map)
	# plt.subplot(1,2,2)
	# plt.imshow(fil.skeleton, cmap='gray')
	# plt.axis('off')
	# plt.show()
	skeleton= fil.skeleton.astype('uint8')
	# print("type of fil skeleton",type(skeleton),skeleton.shape,np.unique(skeleton))
	return skeleton

	
		
	
# def main():                                                                                                                                                                                                         
################################################################################
##print parser arguments
#print("-------------> Dataset:",args.dataset)
#print('-------------> snapshot:',args.snapshot_dir)
#print('-------------> segmentation threshold:',args.segmentation_threshold)
#print('-------------> gaussian threshold:',args.gaussian_threshold)
#print('-------------> backbone :',args.backbone)
################################################################################

#make result directories
result_dir=local_data_path/args.dataset/'results'
result_dir.mkdir(exist_ok=True, parents=True)

#load model
if args.mode=='with_lstm':
	model, lstm_model = load_model()
else:
	model = load_model()

testset= TestDataLoader(dataset=args.dataset,scales=scales,test_dir=args.test_dir)

#construct dataloader
test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,num_workers=1)

eval= EvalOutputs(args)

#enumerate over the DataLoader
for i,batch in enumerate(test_loader):
	# if i<2:
	# 	continue
	print("sample_id: ",i, " out of", len(testset))
	scaled_images,meta= batch

	#perform the prediction scaling
	H,W= meta['image_shape']
	H=H.numpy()
	W=W.numpy()

	max_scale= scales[-1]//4
	scaling_factor_x= W/max_scale
	scaling_factor_y= H/max_scale

	pred_center_line= []
	pred_variance_map= []

	for image in scaled_images:
		image= image.to(device)
		score_map,variance_map, theta_map,backbone_feature= model_inference(model,image,max_scale)
		pred_center_line.append(score_map)
		pred_variance_map.append(variance_map)

	score_map, variance_map = multiscale_aggregation(pred_center_line, pred_variance_map)
	score_map= (score_map*255).astype('uint8')
	#print("before this, the total score map was")
	# print("len of score ")
	# plt.imshow(score_map[0])
	# plt.show()
	component_score_maps, component_center_line = extract_center_line(score_map) 
	# print("total no of predicted contours",len(component_score_maps))
	# for contour in component_score_maps:
	# 	print("debugging individual contours",np.unique(contour))
	# 	plt.imshow(contour)
	# 	plt.show()
	# convolutions=remove_kink(component_score_maps)
	# print("After convolution",np.unique(convolutions[0]))
	# plt.subplot(2,2,1)
	# plt.imshow(component_score_maps[0])
	# plt.subplot(2,2,2)
	# plt.imshow(convolutions[0])
	# plt.subplot(2,2,3)
	# plt.imshow((convolutions[0]>2)*convolutions[0])
	# plt.subplot(2,2,4)
	# plt.imshow((convolutions[0]<=2)*convolutions[0])
	# plt.show()
	print("extract corner ")
	
	
	
	#smoothen_center_line(component_score_maps,corner)
	#apply thresholding to corners 
	

	#segression predicts thicker center lines
	#component score maps contains each of unique contours as an image 
	# i.e. it is a list of images ,containing SKELETONIZED contours 
	#component center line i
	# for index in range(len(component_score_maps)):
		
	# 	#print("trying to check compoennt_score_maps")
	# 	plt.imshow(component_score_maps[index])
	# 	# canvas=np.zeros(component_score_maps.shape)
	# 	# cv2.drawContours(canvas, contours, 0, (255, 255, 255), 1)
	# 	plt.show()
	# for index in range(len(component_score_maps)):
	# 	plt.imshow(component_score_maps[index])
	# 	plt.show()
		# #print("type of compoentn score maps",type(component_score_maps[index]),component_score_maps[index].shape,component_center_line[index].shape)
		# check_continuity(component_score_maps[index],component_center_line[index])
	#print("we are exiting now")
	#component center line is skeletonized contours 
	if args.mode=='with_lstm':
		#print("the code is entering lstm")
		component_score_maps, component_center_line = refine_center_line(backbone_feature,\
													  component_center_line,\
													  lstm_model, max_scale)
		# #print('RECHECK ===============================>')
		# for index in range(len(component_score_maps)):
		# 	plt.imshow(component_score_maps[index])
		# 	plt.show()
			# #print("type of compoentn score maps",type(component_score_maps[index]),component_score_maps[index].shape,component_center_line[index].shape)
			# check_continuity(component_score_maps[index],component_center_line[index])

		
	##print("finished")
	#print("component_score_maps",len(component_score_maps))
	#print("component_center_line",len(component_center_line))

	image_path= meta['image_path'][0]
	image_id= image_path.split('/')[-1].split('.')[0]

	#perform gaussian prediction for each component
	visual_image = np.zeros((score_map[0].shape),dtype='uint8')
	# print("before calling constrict contours")
	# plt.imshow(component_score_maps[0])
	# plt.show()

	sample_contours, center_line_list = construct_contours(variance_map,\
									     component_score_maps,\
									     theta_map)

	#print(len(sample_contours),len(component_score_maps))

	filtered_contour_list=sample_contours
	filtered_center_line_list= center_line_list
	
	filtered_contour_list, filtered_center_line_list = scaling_to_original_scale(scaling_factor_x,\
														scaling_factor_y,filtered_contour_list,\
														filtered_center_line_list)

	##print(" filtered contour shape",filtered_contour_list[0].shape)
	#print(" meta",meta["image_id"])

	#create dataset eval
	#print("calling evaluation code upon the dataset")
	eval.generate_predictions(filtered_contour_list,meta["image_id"][0],filtered_center_line_list)