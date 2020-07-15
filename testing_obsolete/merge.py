import os, glob
import numpy as np 
from shapely.geometry import Polygon
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import shutil as sh
#import geopandas as gpd

image_path='./data/total-text/Images/Test'
write_path= 'results/merging/'
prediction_path= 'results/merged_prediction/'

def crawl_prediction_files(path):

	# description :
	#		Crawl all dump text prediction files names in a directory
	# arguments:
	# 		path : the path of the dump predictions
	#
	# return:
	#		file_list: the list of the files 

	file_list =glob.glob(os.path.join(path,'*.txt'))
	return file_list

def read_prediction(filename):
	# description :
	#		read all dumped prediction 
	# arguments :
	#		filename : the name of the prediction file
	#
	# return :
	#		prediction : the dictonary of polygons in a prediction file : (points, area) 

	# Creating an empty dict 
	prediction_list =  []

	with open(filename) as f:
		content = f.readlines()
	for index, polygon in enumerate(content):
		polygon = np.asarray(polygon.strip().split(','),dtype=int)
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
		p2 = Polygon(polygon[0])
		if p1.intersects(p2):
			area.append(p1.intersection(p2).area)
			flag=True
		else:
			area.append(-1)
	area = np.asarray(area)/p1.area
	return area, flag

def find_skeleton(polygon):
	# description :
	#		find the skeleton of the polygon 
	# arguments:
	#		polygon : the bounding box 
	#
	# return:
	#		skeleton : the list of coordinate shows the skeleton

	return 0

def find_tube_width(polygon):
	# description :
	#		find the width of the bounding box 
	# arguments:
	#		polygon : the bounding box 
	#
	# return:
	#		width   : the width of the boundig box 

	return 0


# source : https://stackoverflow.com/questions/40385782/make-a-union-of-polygons-in-geopandas-or-shapely-into-a-single-geometry
def merge_polygon(polygon_list):
	# description :
	#		union of the list of the bounding boxes 
	# arguments:
	#		polygon_list : the list of bounding box 
	#
	# return:
	#		merge_polygon : the union of the all bounding box 

	return 0

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
	poly = p1.intersects(p2)
	x, y = poly.exterior.coords.xy
	x = np.asarray(x)
	y = np.asarray(y)
	poly_points = np.concatenate([x,y])
	poly_points = poly_points.reshape(2,-1)
	poly_points = poly_points.transpose()
	return poly_points 

def validate_merging_criterion(picked_polygon, base_polygon, threshold):
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
	intersected_polygon_skeleton = find_skeleton(intersected_polygon)
	picked_polygon_skeleton = find_skeleton(picked_polygon)
	base_polygon_skeleton = find_skeleton(base_polygon)

	# find the width of the tube of the base polygon 
	base_polygon_tube_width = find_tube_width(base_polygon, base_polygon_skeleton)

	# the length of the intersected polygon 
	picked_polygon_lenght= len(picked_polygon_skeleton)

	# check condition 
	if base_polygon_tube_width/picked_polygon_length > threshold:
		return True
	else:
		return False

def merge_prediction(prediction_list,drop_threshold=0.4, merge_threshold=0.3):
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

	prediction_list = sort_by_area(prediction_list)

	merged_list=[]

	for idx,picked_polygon in enumerate(prediction_list):

		#pick the polygon with largest area first
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

		# else:
		#	fetch intersected polygon index from merged list
		#	intersected_polygon = np.nonzero((intersection_area_list>merge_threshold)*1)
		#	temprary_list=[]
		#	temprary_list.append(picked_polygon)
		#	for index in intesected_polygon:        
		#		if validate_merging_criterion(picked_polygon, merged_list[index][0],merging_threshold):
		#			merged_list[index][1]=0
		#			temprary_list.append(merged_list[index][0])
		#	merged_polygon= merge_polygon(temprary_list)
		#	merged_list.append([merged_polygon,1])

	return merged_list

def dump_prediction_txt(pred_path,prediction_file_name,polygon_list,swap=False):

	fid = open(pred_path, 'a')
	content=''

	for contour in polygon_list:

		# #skip the contours that dont satisfy the area threshold
		# if contour=='':
		# 	continue
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

def visualization(merged_list, prediction_list, prediction_file_name):
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
			
		cv2.drawContours(image_merged_prediction, [polygon], 0, color, 2)
		#polygon=polygon.reshape(-1)
		polygon_list.append(polygon)

	pred_path= prediction_path+ prediction_file_name.split(os.sep)[-1]
	dump_prediction_txt(pred_path,prediction_file_name,polygon_list,swap=False)
		


	for polygon in prediction_list:
		cv2.drawContours(image_orig_prediction, [polygon['points']], 0, (255,0,255),2)
	
	concat = np.concatenate([image, image_orig_prediction,image_merged_prediction ], axis=1)	

	os.system('mkdir '+write_path)
	cv2.imwrite(write_path+image_filename.split(os.sep)[-1],concat)
	

def merging(path, viz=False):
	# description :	
	#		merging base code caller 
	# arguments:
	#		path : the path of the prediction 
	#		viz  : droping polygon threshold ['default': False]       

	file_list = crawl_prediction_files(path)
	for filename in file_list:
		print('filname', filename)
		prediction_list = read_prediction(filename)
		merged_list = merge_prediction(prediction_list)
		if viz==True:
			visualization(merged_list,prediction_list, filename)

if __name__=='__main__':
	path='results/prediction'
	#print(drop_threshold)
	#drop_threshold= []
	merging(path, viz=True)
