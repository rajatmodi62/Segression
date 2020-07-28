import os, glob
import numpy as np 
from shapely.geometry import Polygon
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import shutil as sh
from skimage.morphology import skeletonize
from shapely.ops import cascaded_union
#import geopandas as gpd

image_path='./data/total-text/Images/Test'
write_path= 'results/merging/'
prediction_path= 'results/merged_prediction/'
centerline_path='results/TOTALTEXTgaussian_threshold=0.65_segmentation_threshold=0.8_lstm_threshold=0.7/center_line_maps'

def crawl_prediction_files(path):

	# description :
	#		Crawl all dump text prediction files names in a directory
	# arguments:
	# 		path : the path of the dump predictions
	#
	# return:
	#		file_list: the list of the files 
	#print(path)
	file_list =glob.glob(os.path.join(path,'*.txt'))
	#print(file_list)
	return file_list

def read_prediction(filename=None,predictions=None, mode='directory'):
	# description :
	#		read all dumped prediction 
	# arguments :
	#		filename : the name of the prediction file
	#
	# return :
	#		prediction : the dictonary of polygons in a prediction file : (points, area) 

	# Creating an empty dict 
	prediction_list =  []
	if mode=='directory':
		centerline_filename = os.path.join(centerline_path,filename.split(os.sep)[-1]) 

		with open(centerline_filename) as f:
			centerline_content = f.readlines()
		
		with open(filename) as f:
			content = f.readlines()
		
		for index, polygon in enumerate(content):
			centerline = np.asarray(centerline_content[index].strip().split(','),dtype=int)
			length = len(centerline)//2
			reduce_pixel = int(0.7*length)
			center_pixel_index = length//2
			centerline = centerline.reshape(-1,2)
			centerline = centerline[center_pixel_index-(reduce_pixel//2):center_pixel_index+(reduce_pixel//2),:]

			#print(centerline)
			polygon = np.asarray(polygon.strip().split(','),dtype=int)
			polygon_area = Polygon(polygon.reshape(-1,2)).area
			element= { 'points': polygon.reshape(-1,2),\
					   'centerline':centerline,\
						'area': polygon_area}
			prediction_list.append(element)
	else:
		for polygon in predictions:
			polygon_area = Polygon(polygon.reshape(-1,2)).area
			element= { 'points': polygon.reshape(-1,2),\
						'area': polygon_area}
			prediction_list.append(element)
	return prediction_list

def sort_by_area(prediction):
	# description :
	#		sort the bounding box according to the area 
	# arguments :
	#		prediction: dictionary of polygon 
	#
	# return : 
	#		sorted_list : list polygon points

	sorted_list= sorted(prediction, key = lambda i: i['area'], reverse= True)
	return sorted_list

def intersection_area_with_base_polygon(merged_list, picked_polygon):
	# description :
	#		find the area of the intersection of the candidate bounding box with respect to the 
	#		merged bounding boxes 
	# argument :
	#		merged_list : the list of merged prediction 
	#		picked_polygon:the merging candidate bounding box 
	#
	# return :
	#		area:the list of area, computed by (area_of_intersection)/(area_of_candidate_bounding_box)
	#		flag:flag to show that the bounding box is not intersecting with any bounding box (T/F)

	area=[]
	flag=False
	p1 = Polygon(picked_polygon)
	for polygon in merged_list:
		if polygon[1]==1 and p1.is_valid:
			p2 = Polygon(polygon[0])
			if p2.is_valid and p1.intersects(p2):
				
				area.append(p1.intersection(p2).area)
				flag=True
			else:
				area.append(-1)
		else:
			area.append(-1)
	area = np.asarray(area)/p1.area
	return area, flag

def find_skeleton_old(polygon):
	# description :
	#		find the skeleton of the polygon 
	# arguments:
	#		polygon : the bounding box 
	#
	# return:
	#		skeleton : the list of coordinate shows the skeleton

	x_max=np.max(polygon[:,0])
	y_max=np.max(polygon[:,1])
	image = np.zeros((int(y_max),int(x_max),3),dtype='uint8')
	polygon = polygon.astype(int)
	cv2.drawContours(image, [polygon], 0, (255,255,255), -1)
	
	image = (image>125)*1
	skeleton = skeletonize(image[:,:,0])
	'''
	plt.subplot(1,2,1)
	plt.imshow(image[:,:,0])
	plt.subplot(1,2,2)
	plt.imshow(skeleton)
	plt.show()
	'''
	#skeleton is an image, extract non zero points from it
	#non_zero_arrays=np.nonzero(skeleton)
	#points=np.concatenate([non_zero_arrays[0],non_zero_arrays[1]]).reshape(2,-1).transpose()
	#return points
	return skeleton

def find_skeleton(polygon, canvas_size):
	# description :
	#		find the skeleton of the polygon 
	# arguments:
	#		polygon : the bounding box 
	#
	# return:
	#		skeleton : the list of coordinate shows the skeleton

	x_max, y_max = canvas_size
	image = np.zeros((int(y_max),int(x_max),3),dtype='uint8')
	polygon = polygon.astype(int)
	cv2.drawContours(image, [polygon], 0, (255,255,255), -1)
	
	image = (image>125)*1
	skeleton = skeletonize(image[:,:,0])
	'''
	plt.subplot(1,2,1)
	plt.imshow(image[:,:,0])
	plt.subplot(1,2,2)
	plt.imshow(skeleton)
	plt.show()
	'''

	return skeleton


def compute_radius(polygon, sampled_points):
	# description :
	#		find the skeleton of the polygon 
	# arguments:
	#		polygon : the bounding box 
	#
	# return:
	#		skeleton : the list of coordinate shows the skeleton

	# polygon : Nx2
	# sampled_points : 3 x 2
	N = polygon.shape[0]
	polygon= np.repeat(polygon[np.newaxis,:,:], 3, axis=0)
	sampled_points= np.repeat(sampled_points[:,np.newaxis,:], N, axis=1)
	
	# polygon --> 3 x N x 2
	# sampled_points ---> 3 x N x 2

	distance = np.power(polygon-sampled_points,2)	# 3 x N x 2
	distance = np.sqrt(np.sum(distance, axis=-1))# 3xN
	distance =np.min(distance, axis=-1) # 3
	radius = np.mean(distance)
	return radius 
	
def find_tube_width(polygon, skeleton, n_center_points= 3):
	# description :
	#		find the width of the bounding box 
	# arguments:
	#		polygon : the bounding box 
	#
	# return:
	#		tube_width   : the width of the boundig box 
	
	
	#skeleton= np.asarray(skeleton)
	#take total no of center points 
	total_center_points= skeleton.shape[0]
	points_indices= np.linspace(0, total_center_points, num=n_center_points+2)#

	#forget start and end point 
	points_indices=points_indices[1:-1]	# 3points 
	points_indices=points_indices.astype(int)
	sampled_point=skeleton[points_indices,:]# 3x2

	radius = compute_radius(polygon, sampled_point)
	tube_width = 2*radius

	return tube_width 

def convert_polygon_to_points(poly):
	# description :
	#		convert shapley polygon object to points 
	# arguments:
	#		poly : shapely polygon object 
	#
	# return:
	#		polygon : the bounding box 
	x, y = poly.exterior.coords.xy
	x = np.asarray(x)
	y = np.asarray(y)
	poly_points = np.concatenate([x,y])
	poly_points = poly_points.reshape(2,-1)
	poly_points = poly_points.transpose()
	return poly_points 

# source : https://stackoverflow.com/questions/40385782/make-a-union-of-polygons-in-geopandas-or-shapely-into-a-single-geometry
def merge_polygon(polygon_list):
	#assume  that all the polygons in the polygon list are already intersecting. 

	# description :
	#		union of the list of the bounding boxes 
	# arguments:
	#		polygon_list : the list of bounding box 
	#
	# return:
	#		merge_polygon : the union of the all bounding box 
	#print('@@@@@@@@@@@@@@@@@@@@@@@@@@ perform merging @@@@@@@@@@@@@@@@@@@@@@@@@@@@@', len(polygon_list))
	poly = cascaded_union(polygon_list)
	polygon = convert_polygon_to_points(poly)
	return polygon 



def intersect_polygon_points(picked_polygon, base_polygon):
	# description :
	#		find the intersection polygon created from two polygon
	# argument :
	#		base_polygon   : the bounding box is already settled 
	#		picked_polygon : the merging candidate bounding box 
	#
	# return :
	#		poly_points   : the intersected bounding box

	p1 = Polygon(picked_polygon)
	p2 = Polygon(base_polygon)
	poly = p1.intersection(p2)
	poly_points = convert_polygon_to_points(poly)
	return poly_points 


def canvas_size(picked_polygon, base_polygon, union_polygon):

	x_max_picked_polygon=np.max(picked_polygon[:,0])
	y_max_picked_polygon=np.max(picked_polygon[:,1])

	x_max_base_polygon=np.max(base_polygon[:,0])
	y_max_base_polygon=np.max(base_polygon[:,1])

	x_max_union_polygon=np.max(union_polygon[:,0])
	y_max_union_polygon=np.max(union_polygon[:,1])


	x_max= max(x_max_picked_polygon,x_max_base_polygon,x_max_union_polygon)
	y_max= max(y_max_picked_polygon,y_max_base_polygon,y_max_union_polygon)
	return x_max, y_max 

def validate_merging_criterion_v1(picked_polygon, base_polygon,  threshold):
	# description :
	#		check the criterion for the merging of the polygon 
	# argument :
	#		base_polygon   : the bounding box is already settled 
	#		picked_polygon : the merging candidate bounding box
	#		threshold      : the threshold for validating the merging box criterion 
	#
	# return :
	#		boolean value shows the validity 

	# compute points of intersected polygon 
	intersected_polygon = intersect_polygon_points(picked_polygon, base_polygon)

	# find the skeleton of the polygon 


	picked_polygon_skeleton = find_skeleton(picked_polygon)

	base_polygon_skeleton = find_skeleton(base_polygon)

	intersected_polygon_skeleton = find_skeleton(intersected_polygon)

	# find the width of the tube of the base polygon 
	base_polygon_tube_width = find_tube_width(base_polygon, base_polygon_skeleton)
	#print('tube width',base_polygon_tube_width)

	# the length of the intersected polygon 
	intersected_polygon_length= len(intersected_polygon_skeleton)

	# check condition 

	score = intersected_polygon_length/base_polygon_tube_width
	#print('----------------->',base_polygon_tube_width, intersected_polygon_length, score)
	if score > threshold and score<1:
		return True
	else:
		return False

def validate_merging_criterion_v2(picked_polygon, base_polygon,  threshold):
	# description :
	#		check the criterion for the merging of the polygon 
	# argument :
	#		base_polygon   : the bounding box is already settled 
	#		picked_polygon : the merging candidate bounding box
	#		threshold      : the threshold for validating the merging box criterion 
	#
	# return :
	#		boolean value shows the validity 



	x_max,y_max = canvas_size(picked_polygon, base_polygon)

	picked_polygon_skeleton = find_skeleton(picked_polygon,[x_max,y_max])

	base_polygon_skeleton = find_skeleton(base_polygon,[x_max,y_max])

	intersected_polygon_skeleton = picked_polygon_skeleton*base_polygon_skeleton
	'''
	plt.subplot(1,3,1)
	plt.imshow(picked_polygon_skeleton)
	plt.subplot(1,3,2)
	plt.imshow(base_polygon_skeleton)
	plt.subplot(1,3,3)
	plt.imshow(intersected_polygon_skeleton)

	plt.show()
	'''

	# the length of the intersected polygon 
	intersected_polygon_length= np.sum(intersected_polygon_skeleton)

	#print('--------------------------->', intersected_polygon_length)

	if intersected_polygon_length > threshold:
		return True
	else:
		return False

def validate_merging_criterion_v3(picked_polygon, picked_centerline, base_polygon,  threshold):
	# description :
	#		check the criterion for the merging of the polygon 
	# argument :
	#		base_polygon   : the bounding box is already settled 
	#		picked_polygon : the merging candidate bounding box
	#		threshold      : the threshold for validating the merging box criterion 
	#
	# return :
	#		boolean value shows the validity 

	union_polygon = merge_polygon([Polygon(picked_polygon), Polygon(base_polygon)])

	x_max,y_max = canvas_size(picked_polygon, base_polygon, union_polygon)

	picked_polygon_skeleton = find_skeleton(picked_polygon,[x_max,y_max])

	base_polygon_skeleton = find_skeleton(base_polygon,[x_max,y_max])

	union_polygon_skeleton = find_skeleton(union_polygon,[x_max,y_max])

	intersection_skeleton_pixels = np.sum(union_polygon_skeleton*base_polygon_skeleton)
	base_skeleton_pixles=np.sum(base_polygon_skeleton)

	'''
	plt.subplot(1,4,1)
	plt.imshow(picked_polygon_skeleton)
	plt.subplot(1,4,2)
	plt.imshow(base_polygon_skeleton)
	plt.subplot(1,4,3)
	plt.imshow(union_polygon_skeleton)
	plt.subplot(1,4,4)
	plt.imshow(picked_polygon_skeleton + base_polygon_skeleton+  union_polygon_skeleton)
	plt.show()
	'''

	#print('--------------------------->', score)

	if score > threshold:
		return True
	else:
		return False

from shapely.geometry import Point

def validate_merging_criterion(picked_polygon, picked_centerline, base_polygon,  threshold):
	# description :
	#		check the criterion for the merging of the polygon 
	# argument :
	#		base_polygon   : the bounding box is already settled 
	#		picked_polygon : the merging candidate bounding box
	#		threshold      : the threshold for validating the merging box criterion 
	#
	# return :
	#		boolean value shows the validity 

	# check how many number of centerline pixels fall within the base polygon 

	
	base_polygon = Polygon(base_polygon)
	pixel_inside=0
	for index in range(picked_centerline.shape[0]):
		point = Point(picked_centerline[index,0], picked_centerline[index,1])
		if base_polygon.contains(point):
			pixel_inside+=1
	score = pixel_inside
	#print('--------------------------->', score)

	if score > threshold:
		return True
	else:
		return False

def merge_centerline(centerline1,centerline2):
	
	return 0





def merge_prediction(prediction_list,variance_map=None, drop_threshold=0.3, merge_threshold=0.3, tube_width_threshold=1):
	# description :
	#		Merging predictions 
	# arguments:
	#
	#		prediction_list : the list of predictions 
	#		drop_threshold  : droping polygon threshold [default : 0.8]
	#		merge_threshold : merging polygon threshold [default : 0.6]	        
	#
	# return:
	#
	#		marged_list:the list of merged polygon [polygon, validity (1/0)]
	#  0-----------Merge_threshold --------------Drop_threshold --------------1
	#  ----- as such -------------|-----merging ---------------|--- droping --| 
	# 
	prediction_list = sort_by_area(prediction_list)

	merged_list=[]

	for idx,picked_polygon in enumerate(prediction_list):

		#pick the polygon with largest area first
		#print('----------_>',picked_polygon.keys())
		picked_centerline = picked_polygon['centerline']
		picked_polygon = picked_polygon['points']


		intersection_area_list, flag =intersection_area_with_base_polygon(merged_list, picked_polygon)

		if flag==False:
			merged_list.append([picked_polygon,1])
			continue

		max_area_intersection_index = np.argmax(intersection_area_list)

		###################################################################################
		#case 1 : check wheather picked polygon is inside the merged polygon list polygon 
		#       : simply drop the picked polygon 
		###################################################################################
                

		if intersection_area_list[max_area_intersection_index]>=drop_threshold:
			continue

		if intersection_area_list[max_area_intersection_index]<merge_threshold:
			merged_list.append([picked_polygon,1])
			continue

		##########################################################################################
		#case 2 : if intersection lies within a threshold 
		#         : find the intersection polygon points 
		#         : find the skeleton of the intersected polygon [s0]
		#         : find the skeleton of the base polygon [s1] (taken from the merged list) and pick polygon[s2]
		#         : find the tube width of base polygon and pick polygon 
		#         :               [using the three ranoom sampled sekeleton points and find with the help of variance ]
		#         : if computed tube width is within the range of threshold then merge the pick polygon with the base polygon 
		###########################################################################################
		
		else:

			#fetch intersected polygon index from merged list
			intersected_polygon = np.nonzero((intersection_area_list>merge_threshold)*1)[0].tolist()

			# for collecting all valid boxes 
			temprary_list=[]
			# merged_centerline=picked_centerline

			# add the picked polygon first 
			temprary_list.append(Polygon(picked_polygon))

			for i in range(len(intersected_polygon)):  
				index=int(intersected_polygon[i])
				if validate_merging_criterion(picked_polygon, picked_centerline, \
				merged_list[index][0],  threshold=tube_width_threshold):
					merged_list[index][1]=0
					temprary_list.append(Polygon(merged_list[index][0]))
					#centerline = merge_centerline(merged_centerline,merged_list[index][2])
			merged_polygon= merge_polygon(temprary_list)
			merged_list.append([merged_polygon,1])

	return merged_list

def dump_prediction_txt(pred_path,prediction_file_name,polygon_list,swap=False):
	# description :
	#				Dump the prediction on the text file 
	# arguments:
	#		pred_path		      : The prediction path
	#		polygon_list    	  : The list of the bounding box 
	#		prediction_file_name  : the filename of the image 	
	#		swap				  : the dumping order [ False: (x,y), True: (y,x) ]

	fid = open(pred_path, 'a')
	content=''

	for contour in polygon_list:
		rows,cols= contour.shape
		for row in range(rows):

			item=contour[row,:]
			x= item[0]
			y= item[1]
			if swap:
				content+=str(y)+','+str(x)+','
			else:
				content+=str(x)+','+str(y)+','
		#remove last comma

		content= content[:-1]+'\n'
		fid.write(content)
		#reset
		content=''
	fid.close()

def visualization(merged_list, prediction_list, prediction_file_name, dumping=False):
	# description :
	#		code to visualize the efect of the merging 
	# arguments:
	#		merged_list          : the list after merging 
	#		prediction_list      : the list of predictions 
	#		prediction_file_name : the filename of the image 	        

	image_filename = os.path.join(image_path,\
					prediction_file_name.split(os.sep)[-1][4:][:-4]+'.jpg') 
	image = cv2.imread(image_filename)
	image_orig_prediction = image.copy()
	image_merged_prediction = image.copy()
	os.system('mkdir '+prediction_path)

	polygon_list=[]
	for polygon, validity in merged_list:
		if validity==0:
			color=(0,255,255)
		else:
			color=(255,255,0)
		polygon=polygon.astype(int)
		#print(polygon.shape)
		cv2.drawContours(image_merged_prediction, [polygon], 0, color, 2)

		polygon_list.append(polygon)

	for polygon in prediction_list:
		cv2.drawContours(image_orig_prediction, [polygon['points']], 0, (255,0,255),2)
	
	concat = np.concatenate([image, image_orig_prediction,image_merged_prediction ], axis=1)	

	os.system('mkdir '+write_path)
	cv2.imwrite(write_path+image_filename.split(os.sep)[-1],concat)

	if dumping:
		pred_path= prediction_path+ prediction_file_name.split(os.sep)[-1]
		dump_prediction_txt(pred_path,prediction_file_name,polygon_list,swap=False)
	

def merging(path, viz=False):
	# description :	
	#		merging base code caller 
	# arguments:
	#		path : the path of the prediction 
	#		viz  : droping polygon threshold ['default': False]       

	file_list = crawl_prediction_files(path)
	for filename in file_list:
		print('filname', filename)
		prediction_list = read_prediction(filename=filename)
		merged_list = merge_prediction(prediction_list)
		if viz==True:
			visualization(merged_list,prediction_list, filename,dumping=True)
		#input('halt')

def extract_valid_polygon(merged_list):
	# description :	
	#				extract valid polygon from all predicted polygon	
	# arguments:
	#		merged_list	: the list of merged polygons contains [polygon, valid_flag] 
	# return:
	#   
	#		final_list	: the list having only valid polygons     

	final_list=[]
	for polygon, valid in merged_list:
		if valid==1:
			final_list.append(polygon)
	return final_list

def merger(prediction_list, filename, viz=False):
	# description :	
	#				wrapper for droping and merging the predictions 
	# arguments:
	#		prediction_list : the list of bounding box 
	#		filename 		: the name of the testig file  
	#		viz    			: flag to set for dumping the visualiziation   
	# return:
	#   
	#		final_list		: the list of valid bounding box 


	prediction_list = read_prediction(predictions=prediction_list,mode='list')
	merged_list = merge_prediction(prediction_list)
	final_list = extract_valid_polygon(merged_list)
	if viz:
		visualization(merged_list,prediction_list, filename,dumping=True)
	return final_list


if __name__=='__main__':
	path='results/TOTALTEXTgaussian_threshold=0.65_segmentation_threshold=0.8_lstm_threshold=0.7/gaussian_maps'
	print("hello")
	#print(drop_threshold)
	#drop_threshold= []
	merging(path, viz=True)

