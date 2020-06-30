import copy
import cv2
import os
import torch.utils.data as data
import scipy.io as io
import numpy as np
from PIL import Image
from util.config import config as cfg
from skimage.draw import polygon as drawpoly
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin
import matplotlib.pyplot as plt
import glob as glob

def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []
        #points list given
        # the area of the polygon should not change much after removing them,
        #the nition is to chosse points in the lines which are not that much close

        #print("points",type(points),len(points))
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)
        #we now have a o=polygon being represented by a set of text instanes
        #{v0 v1 v2 ....vn}

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        #print("splitting the edge into ", n_disk," parts")

        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge
        #since the polygon is being considered in the vector format, we need the  parallel points to be pplaced together

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius
        #print("for the disks radii are",radii)
        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()

        self.transform = transform
        self.texture_list = glob.glob("data/dtd/images/*/*.jpg")
    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4: # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon

    def make_text_region(self, image, polygons):
        h,w= image.shape[:2]
        tr_mask = np.zeros((h//4,w//4), np.uint8)
        train_mask = np.ones((h//4,w//4), np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)//4], color=(1,))
            if polygon.text == '#':
                #print("=================================DATALOADER dontcare")
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)//4], color=(0,))
        return tr_mask, train_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """

        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(cfg.input_size, cfg.input_size))

        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, radius, \
                              tcl_mask, radius_map, sin_map, cos_map, center_line_map,expand=0.3, shrink=2):
        # TODO: shrink 1/2 * radius at two line end
        #print("man, now we are gonna make the centre line in the polygon",len(center_line))
        points_stack=[]
        shrinked_points_stack=[]
        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.asarray([p1, p2, p3, p4])
            #print("before polygon",polygon)
            #print("after polygon",polygon//4)
            #print("tcl mask",tcl_mask.shape,center_line_map.shape)
            self.fill_polygon(tcl_mask, polygon, value=1)
            #print("polygon shape",polygon.shape)
            #fill the center line map with 1/4 resolution
            #print('polygon  --->',polygon)
            polygon_points=polygon.astype(int).copy()
            polygon_points_shrinked = polygon_points//4


            #print('polygon points --->',polygon_points)
            #cv2.drawContours(center_line_map,[polygon_points],-1,(1,1,1),-1)
            points_stack.append(polygon_points)
            shrinked_points_stack.append(polygon_points_shrinked)

            #points_stack.append(np.asarray([p1, p2, p3, p4])//4)
            #self.fill_polygon(np.zeros((128,128)),polygon//4,value=1)
            #self.fill_polygon(radius_map, polygon, value=radius[i])
            #self.fill_polygon(sin_map, polygon, value=sin_theta)
            #self.fill_polygon(cos_map, polygon, value=cos_theta)
        return [points_stack, shrinked_points_stack]

    def create_compressed_gt(self,polygons,center_line_map, type='contour_border', viz=False):

        gt= np.zeros((128,128), np.uint8)
        three_class = np.zeros((3,128,128), np.uint8)

        polygon_edges= np.zeros((128,128), np.uint8)
        contour_edges= np.zeros((128,128), np.uint8)
        for i, polygon in enumerate(polygons):
            temp_edges= np.zeros((128,128), np.uint8)
            if polygon.text != '#':
                point_list= polygon.points.astype(int)//4
                #print("in draw contour",point_list.shape)
                cv2.drawContours(gt,[point_list],-1,(1,1,1),-1)
                cv2.drawContours(temp_edges,[point_list],-1,(1,1,1),-1)
                sobelx = cv2.Sobel(temp_edges,cv2.CV_64F,1,0,ksize=3)
                sobely = cv2.Sobel(temp_edges,cv2.CV_64F,0,1,ksize=3)
                mag = np.sqrt(sobelx**2+sobely**2)
                polygon_edges = polygon_edges + mag
        sobelx = cv2.Sobel(gt,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(gt,cv2.CV_64F,0,1,ksize=3)
        contour_edges = np.sqrt(sobelx**2+sobely**2)
        polygon_edges = (polygon_edges>0.5)*1.0
        contour_edges = (contour_edges>0.5)*1.0
        contour_edges=contour_edges.astype('float32')
        polygon_edges=polygon_edges.astype('float32')
        #center_line_map = cv2.resize(center_line_map, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        temp = gt + center_line_map
        three_class[0,...]= (temp==0)*1.0
        three_class[1,...]= (temp==1)*1.0
        three_class[2,...]= (temp==2)*1.0

        if viz==True:
            #print("godzilla",type(point_list[0]),point_list[0],\
            #np.unique(polygon_edges), np.unique(contour_edges))
            plt.subplot(1,3,1)
            plt.imshow(three_class[0,...])
            plt.subplot(1,3,2)
            plt.imshow(three_class[1,...])
            plt.subplot(1,3,3)
            plt.imshow(three_class[2,...])
            plt.show()
        # if type=='border':
        #     return polygon_edges
        # elif type=='contour_border':
        #     return contour_edges
        # else:
        return [gt,polygon_edges, contour_edges, three_class]

    def get_training_data(self, image, polygons, image_id, image_path,gt_mat_path):
        #print("during entry image shape is",image.shape)
        shape= image.shape
        #print("len",len(shape))
        if len(shape)==2:
            H, W = image.shape
            #copy the image along the 3 dimensions
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            #print("image shape after",image.shape)
        else:
            H,W,_= image.shape
        #print("--------------------------------------------------------------")
        #print("Image shape is",H,W)
        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        if self.transform:
            original_polygons=polygons
            #print("before transformation",image.shape)
            #print("image.shape",image.shape)
            image, polygons = self.transform(image, copy.copy(polygons))
            #print("after transformation",image.shape)
        # compressed_groud_truth= self.create_compressed_gt(polygons)
        #print(" polygons",type(polygons))
        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        #print("tcl_mask_shape",tcl_mask.shape)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)
        center_line_map= np.zeros((image.shape[-3]//4,image.shape[-2]//4) , np.float32)
        # center_line_contour= np.zeros((image.shape[-3]//4,image.shape[-2]//4) , np.float32)
        center_line_contour= np.zeros((image.shape[-3],image.shape[-2]) , np.float32)
        center_line_shrinked_contour= np.zeros((image.shape[-3]//4,image.shape[-2]//4) , np.float32)

        #check the number of disks which are to be fit
        #print("no of disks",cfg.n_disk)
        #print("no of polygons",len(polygons))
        #print(' gt mat path',gt_mat_path)
        center_points_list= []
        radius_list= []
        point_stack=[]
        point_stack_shrinked=[]
        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover(n_disk=cfg.n_disk)

                #now we know the head,tail of snake
                #two edges divided into disk point
                # sideline is the list of divided points on those lines
                # centre is the centre of the disk, and radius computed by the norm
                # what is left now is the step to make the center line of the text
                #print("type of center points",type(center_points))
                #print(center_points)
                center_points_list.append(center_points.astype(int))
                radius_list.append(radius.astype(int))
                #print("godzilla",radius)
                #print("radius",type(radius),radius.shape)
                #print("center points",type(center_points),center_points.shape)
                contour=self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map, cos_map,center_line_map)
                point_stack = point_stack+contour[0]
                point_stack_shrinked = point_stack_shrinked+ contour[1]
        cv2.drawContours(center_line_contour,point_stack,-1,(1,1,1),-1)
        cv2.drawContours(center_line_shrinked_contour,point_stack_shrinked,-1,(1,1,1),-1)

        #print('generate')
        # plt.imshow(center_line_contour)
        # plt.show()
        compressed_groud_truth= self.create_compressed_gt(polygons,center_line_shrinked_contour)

        #print(point_stack) #point_stack =
        #cv2.drawContours(center_line_contour,point_stack,-1,(100,100,100),5)
        #print("after filling up polygon radius map",len(np.unique(radius_map)))
        tr_mask, train_mask = self.make_text_region(image, polygons)
        # plt.imshow(center_line_map)
        # plt.show()
        #print("shapee of center map",center_line_map.shape)
        #print("type of ceentre_points_list",center_points_list[0])
        #print(center_points_list.shape)
        # print(center_points_list)
        center_points_list = np.asarray(center_points_list)//4
        center_points_list.tolist()
        radius_list = np.array(radius_list)//4
        radius_list.tolist()
        #print("len of center point list",len(center_points_list))
        #print("radius shape",radius_list.shape)
        #print("center points list",len(center_points_list))
        #print("len radius list",len(radius_list))
        #print("image id",image_id)
        #draw the images
        # for id in range(len(center_points_list)):
        #     current_contour= center_points_list[id]
        #     current_radius= radius_list[id]
        #     # pick the two values
        #     #take a pair of points
        #     no_of_points= current_contour.shape[0]
        #     for idx in range(no_of_points-1):
        #         #print("h",current_contour[idx:idx+2,:])
        #         #print("current radius",current_radius)
        #         center_line_map= cv2.polylines(center_line_map,[current_contour[idx:idx+2,:]],0,(1,1,1),1)
        #center_line_map=cv2.polylines(center_line_map,center_points_list,0,(1,1,1),1)
        #plt.imshow(center_line_map)
        #plt.show()

        #cv2.drawContours(center_line_map,center_points_list,-1,(1,1,1),0)
        #print("unique in center map", np.unique(center_line_map))
        #cv2.imwrite('test.jpg',center_line_map*255)
        image = image.transpose(2, 0, 1)

        #points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)
        points=[]
        for i, polygon in enumerate(original_polygons):
            pts = polygon.points
            #pts=np.squeeze(pts,axis=0)
            #print("pts",pts.shape)
            points.append(pts)
            #points[i, :pts.shape[0]] = polygon.points
            length[i] = pts.shape[0]
        #print("before meta",image_path)
        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'Height': H,
            'Width': W,
            'mat_path': gt_mat_path,
            'image_shape' : (H,W)
        }
        #print("returning cente rlinemap",type(center_line_map))
        #center_line_map=cv2.resize(center_line_map,(128,128), interpolation=cv2.INTER_NEAREST)
        #train_mask=cv2.resize(train_mask,(128,128), interpolation=cv2.INTER_NEAREST)


        #compressed_groud_truth= cv2.resize(compressed_groud_truth,(128,128), interpolation=cv2.INTER_NEAREST)
        #plt.imshow(compressed_groud_truth)
        #plt.show()
        #center_line_map=center_line_map.unsqueeze(0)
        #print("center_line_map",center_line_map.shape)
        # plt.imshow(compressed_groud_truth)
        # plt.show()
        #train_mask shows the dont care regions in the contours
        # print("train  mask")
        # plt.imshow(train_mask)
        # plt.show()
        #print("centre_line_map shape",np.unique(center_line_map))
        #print("compressed_ground_truth share",compressed_groud_truth.shape,"train mask",train_mask.shape,"tr mask",tr_mask.shape)
        #print("cnter line map",torch.unique(center_line_contour))
        #print("returning")
        return image, compressed_groud_truth,[center_line_contour,center_line_shrinked_contour],train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map#, meta,gt_mat_path

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
