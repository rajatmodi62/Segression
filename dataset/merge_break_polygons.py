import torch
from torch.utils import data, model_zoo
import scipy.io as io
import numpy as np
import os
import cv2 
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
	norm2, vector_cos, vector_sin
from shapely.geometry import LineString
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.blending import aug_image
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from bresenham import bresenham
from shapely.geometry.multipoint import MultiPoint

# skip=0
#pip install bresenham

# Note: This class returns a N//4x N//4 classification of whether center line points are joined or not 
#       Calling function should pass this through the feature extractor backbone.
class MergeBreakPolygons(TextDataset):

	def __init__(self, \
				data_root,\
				input_size=512,\
				ignore_list=None,\
				is_training=True,\
				transform=None):
		super().__init__(transform)
		self.transform = transform
		self.data_root = data_root
		self.input_size=input_size
		self.is_training = is_training

		if ignore_list:
			with open(ignore_list) as f:
				ignore_list = f.readlines()
				ignore_list = [line.strip() for line in ignore_list]
		else:
			ignore_list = []
		self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
		self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
		self.image_list = sorted(os.listdir(self.image_root))
		##print("before filteringimage list",self.image_list[0:5])
		#remove the ignore images from the list
		self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))

		##print("after filtering image list",self.image_list[0:5])
		self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
		#constructed a list containing the dierctory and annottationsin the mat format
		##print(self.annotation_list)
		##print("rajat",len(self.annotation_list))
	def parse_mat(self, mat_path):
		#read the gt_dir mat file 
		# returns:
		#   polygon_list
		"""
		.mat file parser
		:param mat_path: (str), mat file path
		:return: (list), TextInstance
		"""
		##print("matrix parser called")
		##print("",mat_path.split('.')[-1])
		#if mat_path.split('.')[-1]== 'mat':
		if mat_path.split('.')[-1]=='mat':
			##print("true",mat_path)
			annot = io.loadmat(mat_path)
			##print("before annotations",annot)
			polygons = []
			for cell in annot['polygt']:
				x = cell[1][0]
				y = cell[3][0]
				text = cell[4][0] if len(cell[4]) > 0 else '#'
				ori = cell[5][0] if len(cell[5]) > 0 else 'c'

				if len(x) < 4:  # too few points
					continue
				pts = np.stack([x, y]).T.astype(np.int32)
				polygons.append(TextInstance(pts, ori, text))
			##print("type",type(annot['polygt']))
		else:
			##print("in else",mat_path)
			with open(mat_path) as f:
				lines = [line.strip().split(',')[0:8] for line in f.readlines()]
				polygons = []
				for line in lines:
					x=[line[0],line[2],line[4],line[6]]
					y=[line[1],line[3],line[5],line[7]]
					pts = np.stack([x, y]).T.astype(np.int32)
					text= 'dummy'
					ori= 'c'
					polygons.append(TextInstance(pts, ori, text))
		return polygons

	def disk_cover(self, polygon,e1,e2,n_disk=15):
		"""
		cover text region with several disks
		:param n_disk: number of disks
		:return:
		"""
		inner_points1 = split_edge_seqence(polygon, e1, n_disk)
		inner_points2 = split_edge_seqence(polygon, e2, n_disk)
		inner_points2 = inner_points2[::-1]  # innverse one of long edge

		center_points = (inner_points1 + inner_points2) / 2  # disk center
		radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

		return inner_points1, inner_points2, center_points, radii

	def find_canvas(self,polygon_list):
		x_max=0
		y_max=0
		for polygon in polygon_list:
			points=polygon['points']
			rp = polygon['representative_point']
			##print(rp)
			if np.max(points[:,0])+50 > x_max:
				x_max = np.max(points[:,0])+50
			if np.max(points[:,1])+50 > y_max:
				y_max = np.max(points[:,1])+50

		return x_max, y_max

	def visualize(self,polygon_list,center_line=None):
		x_max, y_max = self.find_canvas(polygon_list)
		##print("rajat x,y max",x_max,y_max)
		canvas = np.zeros((int(y_max),int(x_max),3),dtype='uint8')
		for polygon in polygon_list:
			points=polygon['points'].astype(int)
			rp = polygon['representative_point'].astype(int)
			cl = polygon['centerline'].astype(int)
			head_array =polygon['head'].astype(int)
			tail_array=polygon['tail'].astype(int)
			cv2.drawContours(canvas, [points],0,(255,255,255),5)
			cv2.line(canvas, (head_array[0][0],head_array[0][1]), (head_array[1][0],head_array[1][1]), (0, 255, 0), thickness=2)
			cv2.line(canvas, (tail_array[0][0],tail_array[0][1]), (tail_array[1][0],tail_array[1][1]), (255, 255, 0), thickness=2)
			cv2.circle(canvas,(int(rp[0]), int(rp[1])), 5, (0,0,255), -1)
			#draw a circle around center line points 
			##print("center line ",type(cl),cl.shape)
			cv2.circle(canvas,(int(cl[0][0]), int(cl[0][1])), 5, (0, 255, 0), -1)
			cv2.circle(canvas,(int(cl[-1][0]), int(cl[-1][1])), 5, (255, 255, 0), -1)

			#drop the first and last point of the center line while drawing it. 
			#due to the int/float error on border, which creates problem in the intersection heuristic
			for index in range(cl.shape[0]-1):
				if index>1 and index<cl.shape[0]-2:
					cv2.line(canvas, (int(cl[index][0]),int(cl[index][1])), (int(cl[index+1][0]),int(cl[index+1][1])), (255, 255, 255), thickness=2)

		if center_line:
			cv2.line(canvas, (center_line[0][0],center_line[0][1]), (center_line[1][0],center_line[1][1]), (255, 0, 255), thickness=2)
		# plt.imshow(canvas)
		# plt.show()

	def visualize_intersection_polygon(self,polygon_list,intersection_polygon_points,center_line=None):
		x_max, y_max = self.find_canvas(polygon_list)
		canvas = np.zeros((y_max,x_max,3),dtype='uint8')
		for polygon in polygon_list:
			points=polygon['points']
			rp = polygon['representative_point']
			cl = polygon['centerline']
			head_array =polygon['head']
			tail_array=polygon['tail']
			cv2.drawContours(canvas, [points],0,(255,255,255),5)
			cv2.line(canvas, (head_array[0][0],head_array[0][1]), (head_array[1][0],head_array[1][1]), (0, 255, 0), thickness=2)
			cv2.line(canvas, (tail_array[0][0],tail_array[0][1]), (tail_array[1][0],tail_array[1][1]), (255, 255, 0), thickness=2)
			cv2.circle(canvas,(int(rp[0]), int(rp[1])), 5, (0,0,255), -1)
			#draw a circle around center line points 
			cv2.circle(canvas,(int(cl[0][0]), int(cl[0][1])), 5, (0, 255, 0), -1)
			cv2.circle(canvas,(int(cl[-1][0]), int(cl[-1][1])), 5, (255, 255, 0), -1)

			#drop the first and last point of the center line while drawing it. 
			#due to the int/float error on border, which creates problem in the intersection heuristic
			for index in range(cl.shape[0]-1):
				if index>1 and index<cl.shape[0]-2:
					cv2.line(canvas, (int(cl[index][0]),int(cl[index][1])), (int(cl[index+1][0]),int(cl[index+1][1])), (255, 255, 255), thickness=2)
		##print("intersection polygon points",intersection_polygon_points.shape,intersection_polygon_points)
		intersection_polygon_points= intersection_polygon_points.astype('int')

		cv2.drawContours(canvas, [intersection_polygon_points],0,(102,0,51),5)
		if center_line:
			cv2.line(canvas, (center_line[0][0],center_line[0][1]), (center_line[1][0],center_line[1][1]), (255, 0, 255), thickness=2)
		plt.imshow(canvas)
		plt.show()
	
	def visualize_old(self,polygon, image):

		points=polygon['points'].astype('int')
		rp = polygon['representative_point']
		##print(rp)
		
		x_max = abs(np.max(points[:,0])+50)
		y_max = abs(np.max(points[:,1])+50)
		#print("x_max",x_max,"y_max",y_max)
		head_array =polygon['head'].astype('int')
		tail_array=polygon['tail'].astype('int')


		canvas = np.zeros((y_max,x_max,3),dtype='uint8')
		cv2.drawContours(canvas, [points],0,(255,255,255),5)
		cv2.line(canvas, (head_array[0][0],head_array[0][1]), (head_array[1][0],head_array[1][1]), (0, 255, 0), thickness=2)
		cv2.line(canvas, (tail_array[0][0],tail_array[0][1]), (tail_array[1][0],tail_array[1][1]), (255, 255, 0), thickness=2)

		cv2.circle(canvas,(int(rp[0]), int(rp[1])), 5, (0,0,255), -1)
		visual_image= image.copy()
		cv2.drawContours(visual_image, [points],0,(255,0,0),5)
		
		plt.subplot(1,3,1)
		plt.imshow(image)
		plt.subplot(1,3,2)
		plt.imshow(visual_image)
		plt.subplot(1,3,3)
		plt.imshow(canvas)
		plt.show()



	def construct_polygon(self,polygon,mode='object',scale_down=4,viz=False):
		if mode=='numpy':
			points = polygon//scale_down
		else:
			points = polygon.points//scale_down
		##print(polygon.points.shape)
		bottoms = find_bottom(points)  # find two bottoms of this Text
		e1, e2 = find_long_edges(points, bottoms)  # find two long edge sequence
		sideline1, sideline2, center_points, radius  = self.disk_cover(points,e1,e2,n_disk=15)
		number_of_center_points = center_points.shape[0] #nx2
		representative_point = center_points[number_of_center_points//2,:]
		ground_truth = np.ones((number_of_center_points))

		##print('bottom', bottoms[1])
		head =bottoms[1]
		head_array=np.asarray([points[head[0],:],points[head[1],:]])

		tail =bottoms[0]
		tail_array=np.asarray([points[tail[0],:],points[tail[1],:]])

		element={'points':points,\
				'head':tail_array,\
				'tail':head_array,\
				'centerline':center_points,\
				'representative_point': representative_point,\
				'ground_truth':ground_truth,\
				'valid':1 }
		if viz:
			self.visualize(element)
		return element 

	def valid_polygon(self,polygon):
		points = polygon['points']
		# print(points)
		max_coord = np.max(points)
		min_coord = np.min(points)
		# print('max min value', max_coord, min_coord)
		# input('halt')
		if max_coord<128 and min_coord>0:
			return True 
		else:
			return False


	def construct_polygons_list(self,image,polygons_list):
		#description:
		#           construct the e1,e2,upper points, lower points,center line
		#returns:
		#       polygon object
		if self.transform:
			#print("before transformation",image.shape)
			#print("image.shape",image.shape)
			image, polygons_list = self.transform(image, polygons_list)
		#print("after augmentation",polygons)
		polygon_features=[]
		for polygon in polygons_list:
			element =self.construct_polygon(polygon)
			if self.valid_polygon(element):
				polygon_features.append(element)
		# print('number of polygons ', len(polygon_features))
		return image,polygon_features

	def distance(self,point1, point2):
		dist = np.sqrt(np.sum(np.power(point1-point2,2)))
		return dist


	def edge_to_linesegment(self, edge):
		# shapely 
		linesegment = LineString([(edge[0,0],edge[0,1]), (edge[1,0],edge[1,1])])
		return linesegment 
	
	def linesegment_to_numpy(self,linesegment):
		return np.array(linesegment)
	
	def join_center_lines(self,centerline_tail, center_line_head):
		#center_line_head=center_line_head.astype(int)
		#centerline_tail=centerline_tail.astype(int)
		##print('center_line_tail',centerline_tail.shape,centerline_tail)
		#join the center lines to form the linesegment 
		connected_linesegment =  LineString([(center_line_head[0],center_line_head[1]),\
								 (centerline_tail[0],centerline_tail[1])])
		return connected_linesegment 
	
	def number_of_points_in_linesegment(self, linesegment):
		length=int(linesegment.length)
		return length


	def find_line_representative(self,line,midline):
		# line : Nx2 , midline : 2x2	# 1x2
		##print("line 0",line[0],"midline 0",midline[0])
		# if line[0]==midline[0] or line[0]== midline[1]:
		# 	return line[-1]
		# else:
		# 	return line[0]
		
		if np.array_equal(line[0],midline[0]) or np.array_equal(line[0],midline[1]):
			return line[-1]
		else:
			return line[0]

	def reverse_order(self, line):
		new_line = np.zeros((line.shape))
		for index in range(line.shape[0]):
			new_line[line.shape[0]-index-1]=line[index]
		return new_line 

	#------------ T --------------> H
	def reorder(self,line, line_representative,type='head', mode='t2h'):
		##print("line representative",line_representative,line[0],line[-1])
		if mode=='t2h':
			if type=='head':
				if np.array_equal(line_representative,line[0]):
					line = self.reverse_order(line)
			else:
				if np.array_equal(line_representative,line[-1]):
					line = self.reverse_order(line)
		else:
			if type=='head':
				if np.array_equal(line_representative,line[-1]):
					line = self.reverse_order(line)
			else:
				if np.array_equal(line_representative,line[0]):
					line = self.reverse_order(line)
		return line 

	
	def segments_to_point_set(self,linesegment):
		point_set=[]
		##print("godzilla",linesegment.shape)
		for index in range(linesegment.shape[0]-1):
			##print("index",index)
			x_start,y_start=linesegment[index]
			x_end, y_end = linesegment[index+1]
			points = np.array(list(bresenham(int(x_start), int(y_start), int(x_end), int(y_end))))
			##print("babua",points.shape)
			point_set.append(points)
		point_set = np.concatenate(point_set)
		##print("returned")
		return point_set  
	
	def find_order_new(self,merged_center_line,tail,head, viz=False):
		# return a list of order in which the center lines should be merged to be continous in nature 
		# merged_center_line : list of center lines 
		# [centerline1, midline, centerline2]
		# the list consist of three center lines

		head= np.mean(head,axis=0)# head middle point
		tail= np.mean(tail,axis=0)# tail middle point
		line1= merged_center_line[0]
		midline= merged_center_line[1]
		line2= merged_center_line[2]
		line1_representative=self.find_line_representative(line1,midline) # line1 = Nx2 , midline : 2x2	# 1x2
		line2_representative=self.find_line_representative(line2,midline) # line1 = Nx2 , midline : 2x2 #1x2

		# find distance from head --> nearest --> head line , tail_line 

		d1 = self.distance(head,line1_representative)
		d2 = self.distance(tail, line1_representative)
		if d1<d2:
			head_line = line1
			tail_line = line2
			head_line_representative = line1_representative 
			tail_line_representative = line2_representative 
			#order=[2,1,0]
		else:
			head_line = line2
			tail_line = line1
			tail_line_representative = line1_representative 
			head_line_representative = line2_representative 
			#order=[0,1,2]

		# reorder the point sequence (1) head_line, tail_line  (tail to head )
		head_line = self.reorder(head_line, head_line_representative,type='head', mode='t2h')		
		tail_line = self.reorder(tail_line, tail_line_representative,type='tail', mode='t2h')

		# create new line 
		#center_line = np.concatenate([tail_line, head_line],axis=0)
		##print('tail computed =============>')
		tail_line = self.segments_to_point_set(tail_line)
		gt_tail_line = np.ones((tail_line.shape[0]),dtype='uint8')
		##print('head computed ==============>')
		head_line = self.segments_to_point_set(head_line)
		gt_head_line = np.ones((head_line.shape[0]),dtype='uint8')
		##print('midline computed ==============>')
		middle_line = np.concatenate([tail_line[-1,:].reshape(-1,2),head_line[0,:].reshape(-1,2)],axis=0)
		##print("modline ki shape",middle_line.shape,tail_line.shape,head_line.shape)
		middle_line = self.segments_to_point_set(middle_line)
		gt_middle_line = np.zeros((middle_line.shape[0]),dtype='uint8')
		
		##print('shapes', tail_line.shape, middle_line.shape, head_line.shape)
		center_line = np.concatenate([tail_line,middle_line, head_line],axis=0)
		ground_truth = np.concatenate([gt_tail_line, gt_middle_line,gt_head_line])

		if viz:
			x_max=np.max(center_line[:,0])+50
			y_max=np.max(center_line[:,1])+50
			canvas = np.zeros((int(y_max),int(x_max),3),dtype='uint8')
			for index in range(tail_line.shape[0]-1):
				cv2.line(canvas, (tail_line[index][0],tail_line[index][1])\
				, (tail_line[index+1][0],tail_line[index+1][1]), (0, 255, 0), thickness=2)	
				# plt.imshow(canvas)
				# plt.show()
				##print("#printing tail line")
				cv2.imshow('image',canvas)
				cv2.waitKey(500)
				cv2.destroyAllWindows()

			for index in range(middle_line.shape[0]-1):
				cv2.line(canvas, (middle_line[index][0],middle_line[index][1])\
				, (middle_line[index+1][0],middle_line[index+1][1]), (255, 255, 0), thickness=2)	
				# plt.imshow(canvas)
				# plt.show()	
				##print("#printing middle line")
				cv2.imshow('image',canvas)
				cv2.waitKey(500)
				cv2.destroyAllWindows()

			for index in range(head_line.shape[0]-1):
				cv2.line(canvas, (head_line[index][0],head_line[index][1])\
				, (head_line[index+1][0],head_line[index+1][1]), (0, 0, 255), thickness=2)
				# plt.imshow(canvas)
				# plt.show()	
				##print("showing head line")
				cv2.imshow('image',canvas)
				cv2.waitKey(500)
				cv2.destroyAllWindows()
		
		return center_line, ground_truth

	def convert_polygon_to_points(self,poly):
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
	def resolve_polygon(self,polygon):
		#4x2
		diag1 = LineString([(polygon[0,0],polygon[0,1]), (polygon[2,0],polygon[2,1])])
		diag2= LineString([(polygon[1,0],polygon[1,1]), (polygon[3,0],polygon[3,1])])
		#print("@@@@@@@@@@@@@@@@@@STATUS",diag1.intersects(diag2))
		if diag1.intersects(diag2):
			return polygon 
		else:
			#input('halt')
			new_polygon = np.concatenate([polygon[0:2,:],\
							  polygon[3:4,:], polygon[2:3,:]],axis=0) #4x2
			#swap second and third point
			# #print("before mergin polygon")
			# #print(polygon) 
			# #print("===============>rajat plotting")
			
			# temp= polygon[1]
			# #print("temp",temp)
			# #print("swap polygon1",polygon[1])
			# polygon[1]=polygon[2]
			# #print("swap polygon1",polygon[1])
			
			# polygon[2]=temp
			# #print("swaap polygon 2",polygon[2])

			# polygon[[1,2]]=polygon[[2,1]]
			polygon[[2,3]]=polygon[[3,2]]
			# #print('after swapping ', polygon)
			diag1 = LineString([(polygon[0,0],polygon[0,1]), (polygon[2,0],polygon[2,1])])
			diag2= LineString([(polygon[1,0],polygon[1,1]), (polygon[3,0],polygon[3,1])])
			# #print("@@@@@@@@@@@@@@@@@@AFTER SWAP",diag1.intersects(diag2))
			# #print("===============>rajat plotting")
			# plt.plot(polygon)
			# plt.show()
			return polygon
	def merge_polygon(self,polygon1,polygon2,intersection_polygon_head_tail,\
		 merged_center_line, unified_polygon_head_tail, centerline_gt):
		# head : 2x2
		# tail : 2x2
		
		# polygon={'points':None,\
		# 'head':None,\
		# 'tail':None,\
		# 'centerline':center_points,\
		# 'representative_point': representative_point,\
		# 'ground_truth':ground_truth,\
		# 'valid':1 }


		polygon={'points':None,\
		'head':None,\
		'tail':None,\
		'centerline':None,\
		'representative_point': None,\
		'ground_truth':None,\
		'valid':1 }

		intersection_polygon_head, intersection_polygon_tail=intersection_polygon_head_tail
		unified_polygon_head, unified_polygon_tail=unified_polygon_head_tail

		#convert linestrings to numpy array
		intersection_polygon_head= self.linesegment_to_numpy(intersection_polygon_head)
		intersection_polygon_tail= self.linesegment_to_numpy(intersection_polygon_tail)
		unified_polygon_head= self.linesegment_to_numpy(unified_polygon_head)
		unified_polygon_tail= self.linesegment_to_numpy(unified_polygon_tail)

		##print("intersection polygon head",type(intersection_polygon_head),intersection_polygon_head.shape)
		##print("------------> before",intersection_polygon_head,intersection_polygon_tail.shape,intersection_polygon_tail[1:2,:].shape,intersection_polygon_tail[0:1,:].shape)
		# intersected_polygon = np.concatenate([intersection_polygon_head,intersection_polygon_tail[1:2,:],\
		# 					  intersection_polygon_tail[0:1,:]],axis=0) #4x2
		# intersected_polygon = np.concatenate([intersection_polygon_head,intersection_polygon_tail[0:1,:],\
							#   intersection_polygon_tail[1:2,:]],axis=0) #4x2
		intersected_polygon = MultiPoint([(intersection_polygon_head[0][0], intersection_polygon_head[0][1]),\
								(intersection_polygon_head[1][0], intersection_polygon_head[1][1]),\
								(intersection_polygon_tail[0][0], intersection_polygon_tail[0][1]),\
								(intersection_polygon_tail[1][0], intersection_polygon_tail[1][1])]).convex_hull
		try:
			pp = self.convert_polygon_to_points(intersected_polygon)
			#print('pp.shape', pp.shape)
			#if pp.shape[0]<4:
				#input('halt')
		except:
			pass
			#print(np.array(intersected_polygon))
			#input('halt')
		##print("intersected polygon",intersected_polygon)
		##print("rajat polygon1",type(polygon1),polygon1,type(polygon2),polygon2)
		#self.visualize_intersection_polygon([polygon1,polygon2],intersected_polygon)
		#intersected_polygon=self.resolve_polygon(intersected_polygon)
		#print("intersected polygon",intersected_polygon)
		#print(polygon1['points'],polygon2['points'],intersected_polygon)
		unified_polygon=cascaded_union([Polygon(polygon1['points']),Polygon(polygon2['points']),Polygon(intersected_polygon)])
		
		#print("type of unified_polygon",type(unified_polygon))
		#print("polygon1",polygon1['points'])
		#print("poygon2",polygon2['points'])
		# if unified_polygon.geom_type == 'MultiPolygon':
		# 	self.visualize_intersection_polygon([polygon1,polygon2],intersected_polygon.astype('int'))
		# else:
		# 	global skip
		# 	skip+=1
		# 	if skip==1:
		# 		self.visualize_intersection_polygon([polygon1,polygon2],intersected_polygon.astype('int'))

		unified_polygon = self.construct_polygon(self.convert_polygon_to_points(unified_polygon),mode='numpy')    
		##print("==================visualizing unified polygon")
		#self.visualize([unified_polygon])

		polygon['points']=unified_polygon['points']
		polygon['head']=unified_polygon['head']
		polygon['tail']=unified_polygon['tail']

		merged_center_line,  centerline_gt = self.find_order_new(merged_center_line,unified_polygon['tail'],unified_polygon['head'])
		##print('order', order)
		#input('halt')

		#centerline_gt=np.concatenate([centerline_gt[order[0]],\
		#centerline_gt[order[1]],centerline_gt[order[2]]],axis=0)
		##print("centerline_gt",centerline_gt.shape, "merged_center_line",merged_center_line.shape)
		
		polygon['centerline']=merged_center_line
		polygon['ground_truth']=centerline_gt
		polygon['representative_point']=unified_polygon['representative_point']
		polygon['valid']=1

		return polygon

	def valid_merge(self, reference_polygon,picked_polygon,proximity_threshold=1000):

		meta={'condition':False,\
			'polygon':None}

		reference_polygon_centerline = reference_polygon['centerline']
		reference_polygon_gt=reference_polygon['ground_truth']
		reference_polygon_centerline_end =reference_polygon['centerline'][-3,:]
		reference_polygon_centerline_start =reference_polygon['centerline'][2,:]
		reference_polygon_head=self.edge_to_linesegment(reference_polygon['head'])
		reference_polygon_tail=self.edge_to_linesegment(reference_polygon['tail'])
		reference_polygon_centerline_end_orig =reference_polygon['centerline'][-1,:]
		reference_polygon_centerline_start_orig =reference_polygon['centerline'][0,:]

		picked_polygon_centerline = picked_polygon['centerline']
		picked_polygon_gt=picked_polygon['ground_truth']
		picked_polygon_centerline_start =picked_polygon['centerline'][2,:]
		picked_polygon_centerline_end =picked_polygon['centerline'][-3,:]
		picked_polygon_head=self.edge_to_linesegment( picked_polygon['head'])
		picked_polygon_tail=self.edge_to_linesegment( picked_polygon['tail'])
		picked_polygon_centerline_start_orig =picked_polygon['centerline'][0,:]
		picked_polygon_centerline_end_orig =picked_polygon['centerline'][-1,:]

		#picking 4 combinatiosn of center line distances
		center_end_point_set=[[reference_polygon_centerline_end,picked_polygon_centerline_start],\
		[reference_polygon_centerline_start, picked_polygon_centerline_end],\
		[reference_polygon_centerline_end, picked_polygon_centerline_end],\
		[reference_polygon_centerline_start, picked_polygon_centerline_start]]

		#picking 4 combinatiosn of center line distances
		center_end_point_set_orignal=[[reference_polygon_centerline_end_orig,picked_polygon_centerline_start_orig],\
		[reference_polygon_centerline_start_orig, picked_polygon_centerline_end_orig],\
		[reference_polygon_centerline_end_orig, picked_polygon_centerline_end_orig],\
		[reference_polygon_centerline_start_orig, picked_polygon_centerline_start_orig]]

		# take pairs of head tails of reference /picked polygon
		edge_point_set=[[reference_polygon_tail,picked_polygon_head],\
		[reference_polygon_head,picked_polygon_tail],\
		[reference_polygon_tail,picked_polygon_tail],\
		[reference_polygon_head,picked_polygon_head]]

		# final  head/tail combinations of the merged polygon 
		unified_end_points=[[reference_polygon_head,picked_polygon_tail],\
		[picked_polygon_head,reference_polygon_tail],\
		[reference_polygon_head,picked_polygon_head],\
		[reference_polygon_tail,picked_polygon_tail]]

		distance=[]
		for index in range(len(center_end_point_set)):
			distance.append(self.distance(center_end_point_set[index][0],center_end_point_set[index][1]))

		distance = np.asarray(distance)
		min_distance=np.argmin(distance)

		#try to form the center line from the side where distance is minimum
		merged_center_line =self.join_center_lines(center_end_point_set[min_distance][0],\
							center_end_point_set[min_distance][1])

		#try to form the center line from the side where distance is minimum
		merged_center_line_orig =self.join_center_lines(center_end_point_set_orignal[min_distance][0],\
								 center_end_point_set_orignal[min_distance][1])

		# #print("rajat merged center line",merged_center_line)
		# #print("bidu visualize kar=======================================")
		
		# #print("---------_>rajat condition",merged_center_line.intersects(edge_point_set[min_distance][0]),\
		# 		merged_center_line.intersects(edge_point_set[min_distance][1]))

		# self.visualize([reference_polygon,picked_polygon],\
		# 			center_line=[center_end_point_set[min_distance][0].astype(int),center_end_point_set[min_distance][1].astype(int)])
		nop=self.number_of_points_in_linesegment(merged_center_line_orig)
		
		if merged_center_line.intersects(edge_point_set[min_distance][0])\
		 	and merged_center_line.intersects(edge_point_set[min_distance][1])\
			and nop<proximity_threshold:
		 	
			##print("bidu, true me hoon")
			meta['condition']= True
			
			merged_center_line_gt = np.zeros((nop))
			# #print("merged center line git",merged_center_line_gt.shape)
			try:
				polygon = self.merge_polygon(reference_polygon,picked_polygon,\
				[edge_point_set[min_distance][0],edge_point_set[min_distance][1]],\
				[reference_polygon_centerline,self.linesegment_to_numpy(merged_center_line_orig),picked_polygon_centerline],\
				[unified_end_points[min_distance][0],unified_end_points[min_distance][1]],\
				[reference_polygon_gt,merged_center_line_gt,picked_polygon_gt])
			except:
				meta['condition']=False

		##print("bidu,false")		
		if meta['condition']:
			meta['polygon']=polygon
			##print("=============NOT visualizing merged polygon")  
			self.visualize([polygon])
		
		return meta


	def find_nearest(self,polygon_list,candidate_polygon_index):
		reference_polygon=polygon_list[candidate_polygon_index]['representative_point']
		nearest_index=candidate_polygon_index
		min_distance=1e7
		for index in range(len(polygon_list)):
			picked_polygon = polygon_list[index]['representative_point']
			picked_polygon_validity=polygon_list[index]['valid']
			if picked_polygon_validity==1:
				picked_polygon = polygon_list[index]['representative_point']
				dist=self.distance(picked_polygon,reference_polygon)
				if dist<min_distance:
					min_distance=dist
					nearest_index=index
		return nearest_index 


	def merge_random_polygons(self,polygon_list_orig,max_length, n_merges):
		# attempt = len(polygon_list_orig)
		attempt=100
		##print("rajat m odi",len(polygon_list_orig),n_merges)
		# desciption:
		#   create a group of n_merges polygons
		new_polygon_list=[]     # list of list [polygon, valid], valid=1/0
		##print('mumber of polygons', len(polygon_list_orig))
		#input('halt')
		
		flag=0
		#print("Attempt",attempt, "n_merges",n_merges)
		for i in range(attempt):
			polygon_list=polygon_list_orig.copy()
			candidate_polygon_index = np.random.randint(0,len(polygon_list))
			polygon_list[candidate_polygon_index]['valid']=0
			#max_merges = np.random.randint(1,n_merges)
			##print("max merges",max_merges)
			reference_polygon = polygon_list[candidate_polygon_index]
			##print('candidate polygon index',candidate_polygon_index)
			times=0
			for index in range(0,n_merges):
				picked_polygon_index  = self.find_nearest(polygon_list,candidate_polygon_index)
				##print('nearest polygon', picked_polygon_index)
				picked_polygon= polygon_list[picked_polygon_index]
				polygon_list[picked_polygon_index]['valid']=0
				#self.visualize([reference_polygon,picked_polygon])
				meta = self.valid_merge(reference_polygon,picked_polygon)
				##print(meta['condition'])
				if meta['condition']:
					times+=1
					#print('============> merged',times,"times")
					reference_polygon =meta['polygon']
					candidate_polygon_index=len(polygon_list)
					polygon_list.append(reference_polygon)      
					reference_polygon = polygon_list[candidate_polygon_index] 
			length = reference_polygon['centerline'].shape[0]
			if length < max_length:
				flag=1
				break;
		if flag==0:
			candidate_polygon_index = np.random.randint(0,len(polygon_list))
			reference_polygon = polygon_list[candidate_polygon_index]
			# update center line and groudntruth 
			reference_polygon['centerline'] = self.segments_to_point_set(reference_polygon['centerline'])
			reference_polygon['ground_truth'] = np.ones((reference_polygon['centerline'].shape[0]),dtype='uint8')
			length = reference_polygon['centerline'].shape[0]
			if length < max_length:
				flag=1
		##print("=======done")
		return reference_polygon, flag
		
		
	def __getitem__(self, item, max_length=500):
		sequence_coordinate=np.zeros((max_length,2),dtype=int)
		ground_truth = np.zeros((max_length),dtype=int)

		# read the image, ground truth 
		image_id = self.image_list[item]
		##print("image_id",image_id)
		image_path = os.path.join(self.image_root, image_id)
		#print(image_id)
		# Read image data
		image = pil_load_img(image_path)
		# plt.imshow(image)
		# plt.show()
		

		#image= aug_image(image,self.texture_list)
		# Read annotation
		annotation_id = self.annotation_list[item]
		#
		##print("annotation_id",annotation_id)
		#correct the annotation_id for txt case of chinese dataset
		if annotation_id.split('.')[0].split('_')[-1]=='synth':
			annotation_id= annotation_id.split('.')[0]
			splits= annotation_id.split('_')
			annotation_id= ''
			for split in splits[2:]:
				annotation_id+= split+'_'
			annotation_id= annotation_id[:-1]+'.txt'
			##print("filtered annotation id ",annotation_id)

			##print("corrected_annotation_id",annotation_id)
		annotation_path = os.path.join(self.annotation_root, annotation_id)

		polygon_list= self.parse_mat(annotation_path)
		original_image=image.copy()
		# for each polygon,apply transform, construct the e1,e2,upper points, lower points,center line
		image,polygons = self.construct_polygons_list(image,polygon_list)
		valid=0
		length=0
		if len(polygons)>0:
			# join the center lines of n_merge polygons. 
			# reference_polygon= self.merge_random_polygons(polygons,n_merges=len(polygons))
			reference_polygon, valid= self.merge_random_polygons(polygons,max_length,n_merges=3)
			length = reference_polygon['centerline'].shape[0]
			sequence_coordinate[0:length,:] = reference_polygon['centerline']
			ground_truth[0:length] = reference_polygon['ground_truth']
			#self.visualize_old(reference_polygon, image)
		image=image.transpose(2,0,1)
		return {'image': image, 'coordinate_sequence':sequence_coordinate, 'gt':ground_truth, 'length':length, 'valid': valid }

	def __len__(self):
		return len(self.image_list)

if __name__ == '__main__':
	import os
	from util.augmentation import BaseTransform, Augmentation

	means = (0.485, 0.456, 0.406)
	stds = (0.229, 0.224, 0.225)

	transform = Augmentation(
		size=512, mean=means, std=stds
	)

	# transform=None
	trainset = MergeBreakPolygons(
		data_root='data/total-text-original',
		# ignore_list='./ignore_list.txt',
		is_training=True,
		transform=transform
	)

	# #print(len(trainset))
	#trainset[238]
	# #print("done!")

	train_loader = data.DataLoader(trainset, batch_size=8, shuffle=True)
	# for idx in range(len(trainset)):
	# 	if idx%200==0:
	# 		print(idx)
	# 	#print("==================================CALLING SAMPLE====================")
	# 	dict=trainset[idx]
	# 	coordinate_sequence=dict['coordinate_sequence']
	# 	valid=dict['valid']
	# 	gt = dict['gt']
	# 	length = dict['length']
	# 	print(coordinate_sequence.shape, valid, gt.shape, length, np.unique(gt))
		# if len(np.unique(gt))==1:
		# 	input('halt')
		#print("==================================sample done!!====================")
	# ##print(type(img),img.shape,train_mask.shape,center_line_map.shape)

	for step, (batch) in enumerate(train_loader):
		print('coordinate_sequence', torch.min(batch['coordinate_sequence']),torch.max(batch['coordinate_sequence']))
		print(step)
		print('valid', batch['valid'].shape)
		print(step,len(batch))