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
from testloader import TestDataLoader
from model.east_gaussian_rotated import EAST
from util.augmentation import BaseTransform
from shapely.geometry import Polygon
from skeletonize import skeletonize_image
import shutil as sh
from pathlib import Path
import math
import torch.nn.functional as F
from PIL import Image, ImageDraw
import scipy.io as io
from util.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin
from model.east_gaussian_rotated import EAST


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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




#define test dataloader
''' INPUT: List of Testing scales'''
''' Returns resized image '''

class TestDataLoader(data.Dataset):

    def __init__(self,max_size, scaling_factor=[1]):
        super().__init__()
        #get the list of image path
        self.test_img_dir= './data/total-text/Images/Test'
        self.test_gt_dir= './data/total-text/gt/Test'
        self.scaling_factor=scaling_factor
        #print(os.listdir(self.test_img_dir))
        self.test_img_path= [os.path.join(self.test_img_dir,path) for path in os.listdir(self.test_img_dir)]
        self.test_gt_path= [os.path.join(self.test_gt_dir,path) for path in os.listdir(self.test_gt_dir)]
        self.means = (0.485, 0.456, 0.406)
        self.stds = (0.229, 0.224, 0.225)
        self.max_size = max_size


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
                              tcl_mask, radius_map, sin_map, cos_map, center_line_map,expand=0.3, shrink=1):

        # TODO: shrink 1/2 * radius at two line end
        #print("man, now we are gonna make the centre line in the polygon",len(center_line))
        points_stack=[]
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
            #self.fill_polygon(tcl_mask, polygon, value=1)
            #print("polygon shape",polygon.shape)
            #fill the center line map with 1/4 resolution
            #print('polygon  --->',polygon)
            polygon_points=  polygon.astype(int).copy()
            polygon_points = polygon_points//4
            #print('polygon points --->',polygon_points)
            #cv2.drawContours(center_line_map,[polygon_points],-1,(1,1,1),-1)
            points_stack.append(polygon_points)
            #points_stack.append(np.asarray([p1, p2, p3, p4])//4)
            #self.fill_polygon(np.zeros((128,128)),polygon//4,value=1)
            #self.fill_polygon(radius_map, polygon, value=radius[i])
            #self.fill_polygon(sin_map, polygon, value=sin_theta)
            #self.fill_polygon(cos_map, polygon, value=cos_theta)
        return points_stack

    def get_center_line(self,image,gt_path,scale=512):
        #print("got the value",gt_path)
        #print('SCALE ==>', scale)
        annot = io.loadmat(gt_path)
        polygons = []
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
            pts = np.stack([(x/image.shape[1])*scale[1], (y/image.shape[0])*scale[0]]).T.astype(np.int32)
            #print(pts, image.shape)
            polygons.append(TextInstance(pts, ori, text))

        for i, polygon in enumerate(polygons):
            #print("now polygon is",polygon,type(polygon))
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        #define the shapes of the masks
        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)
        center_line_map= np.zeros((image.shape[-3]//4,image.shape[-2]//4) , np.float32)
        center_line_contour= np.zeros((scale[0]//4,scale[1]//4) , np.float32)

        center_points_list= []
        radius_list= []
        point_stack=[]
        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover(n_disk=cfg.n_disk)
                center_points_list.append(center_points.astype(int))
                radius_list.append(radius.astype(int))

                contour=self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map, cos_map,center_line_map)
                point_stack = point_stack+contour

        #scale the points in the ground truth
        # for i,points in enumerate(point_stack):
        #     point_stack[i][:,0]= (point_stack[i][:,0]/image.shape[0])*scale
        #     point_stack[i][:,1]= (point_stack[i][:,1]/image.shape[1])*scale

        cv2.drawContours(center_line_contour,point_stack,-1,(1,1,1),-1)
        # plt.imshow(center_line_contour)
        # plt.show()
        return center_line_contour

    def resize_img(self,img,  h,w):
        resize_w = w
        resize_h = h
        resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
        resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
        #print("type of variacles",type(resize_h),type(resize_w), resize_w, resize_h)
        img= cv2.resize(img,(resize_w,resize_h), interpolation=cv2.INTER_NEAREST)
        # print('after modification')
        # plt.imshow(img)
        # plt.show()
        #img = cv2.resize((resize_w, resize_h), Image.BILINEAR)
        ratio_h = resize_h / h
        ratio_w = resize_w / w
        return img, ratio_w, ratio_h

    def reset_image_size(self, H,W, img, gt_path):
        img_list=[]
        scaled_center_line_image=[]
        size_list=[]
        center_line_list=[]
        ratio_list=[]

        for index in range(len(self.scaling_factor)):
            #print(index)
            imgx = img.copy()
            scaling_factor = self.scaling_factor[index]

            H_new = int(H*scaling_factor)
            W_new = int(W*scaling_factor)
            imgx, ratio_w, ratio_h = self.resize_img(imgx, H_new, W_new)
            #print('scaling factor', scaling_factor, 'h', H_new, 'w', W_new)

            transform = BaseTransform(
                size=[imgx.shape[1], imgx.shape[0]], mean=self.means, std=self.stds
            )
            img_list.append(transform(imgx)[0].transpose(2,0,1))
            size_list.append([imgx.shape[0], imgx.shape[1]])
            # plt.imshow(transform(imgx)[0])
            # plt.show()
            scaled_center_line_image.append(self.get_center_line(img,gt_path,[imgx.shape[0], imgx.shape[1]]))
            #img_list.append(imgx)
            ratio_list.append([ratio_w, ratio_h])
        print('amazing', len(img_list))
        return img_list, ratio_list, scaled_center_line_image, size_list

    def __len__(self):
        return len(self.test_img_path)

    def pil_load_img(self, path):
        image = Image.open(path).convert('RGB')
        imagex = self.check_max_size(image)
        image = np.array(image)
        imagex = np.array(imagex)
        print('oringinal image shape wihout alteration ==>',image.shape, imagex.shape)
        return imagex,image

    def check_max_size(self, image):
        W, H = image.size
        max = H
        if max < W:
            max = W
        if max > self.max_size:
            print(' HHHHEEELLLOOO', max, W, H,int((W/max)*self.max_size),int((H/max)*self.max_size) )
            image = image.resize((int((W/max)*self.max_size),int((H/max)*self.max_size)), Image.BICUBIC)
        return image

    def __getitem__(self, idx):
        image, image_orignal = self.pil_load_img(self.test_img_path[idx])
        H_orig, W_orig,_ = image_orignal.shape
        # plt.imshow(image)
        # plt.show()
        img_path=self.test_img_path[idx]
        img_id= img_path.split('/')[-1].split('.')[0]
        gt_path= 'data/total-text/gt/Test/'+'poly_gt_'+img_id+'.mat'
        H, W, _ = image.shape
        scaled_images, ratio_list, scaled_center_line_image, size_list= self.reset_image_size(H,W,image, gt_path)
        image_path= self.test_img_path[idx]
        print("in dataloader",image_path)

        meta = {
            'image_path': image_path,
            'image_shape' : (H_orig,W_orig)
        }
        # self.get_center_line(image,gt_path)
        return scaled_images,scaled_center_line_image,meta, size_list

def area(x, y):
    polygon = Polygon(np.stack([x, y], axis=1))
    #print("shape of stack",np.stack([x, y], axis=1).shape)
    return float(polygon.area)

# def create_gaussian_array(variance_x,variance_y,theta,x,y,height,width):
#     # i varies by height
#     #print("shapes",variance_x.shape,variance_y.shape,theta.shape)
#     sin = math.sin(theta)
#     cos = math.cos(theta)
#     var_x = math.pow(variance_x,2)
#     var_y = math.pow(variance_y,2)
#     a= (cos*cos)/(2*var_x) + (sin*sin)/(2*var_y)
#     b= (-2*sin*cos)/(4*var_x) + (2*sin*cos)/(4*var_y)
#     c= (sin*sin)/(2*var_x) + (cos*cos)/(2*var_y)
#     #
#     i= np.array(np.linspace(0,height-1,height))
#     i= np.expand_dims(i,axis=1)
#     i=np.repeat(i,height,axis=1)
#     j=i.T
#     #print("a",a,b,c )
#     #the x & y around which gaussian is centered
#     A= np.power(i-x,2)
#     # B=1
#     B= 2*(i-x)*(j-y)
#     C= np.power(j-y,2)
#     gaussian_array= np.exp(-(a*A+b*B+c*C))
#     return gaussian_array


def create_gaussian_array(variance_x,variance_y,theta,x,y,height,width):


    sin = math.sin(theta)
    cos = math.cos(theta)
    var_x = math.pow(variance_x,2)
    var_y = math.pow(variance_y,2)
    a= (cos*cos)/(2*var_x) + (sin*sin)/(2*var_y)
    b= (-2*sin*cos)/(4*var_x) + (2*sin*cos)/(4*var_y)
    c= (sin*sin)/(2*var_x) + (cos*cos)/(2*var_y)


    i= np.linspace(0,width-1,width)
    i= np.tile(i,(height,1)).T

    j= np.linspace(0,height-1,height)
    j= np.tile(j,(width,1))
    #the x & y around which gaussian is centered
    A= np.power(i-x,2)
    #print("A shape",A.shape)
    # B=1
    #print("i shape",i.shape,"j shape",j.shape)
    C= np.power(j-y,2)
    #print("C shape",C.shape)
    B= 2*(i-x)*(j-y)
    gaussian_tensor= np.exp(-(a*A+b*B+c*C))
    # print("gaussian tensor shape",gaussian_tensor.shape)
    #
    # plt.imshow(gaussian_tensor)
    # plt.show()
    return gaussian_tensor

def voting(center_line_list, scales):
    #get the minimum no of votes needed
    min_votes_needed= len(scales)//2 + 1
    #perform the sum in the list
    mask = np.zeros(center_line_list[0].shape)

    for scaled_center_line_image in center_line_list:
        mask+= scaled_center_line_image

    #perform voting
    mask= (mask>= min_votes_needed).astype('uint8')
    return mask

def model_forward_pass(model, image,max_scale_index):
    score_map,contour_map, flag,variance_map=model(image)

    score_map= score_map.squeeze().detach().cpu().numpy()
    variance_map= variance_map.squeeze()
    # theta_map=  3.14*F.sigmoid(variance_map[2,:,:])
    # theta_map= theta_map.detach().cpu().numpy()
    variance_map= variance_map.detach().cpu().numpy()

    print("vaariance map shape",variance_map.shape)

    score_map= (score_map>0.40).astype('uint8')
    score_map=cv2.resize(score_map,(int(size_list[max_scale_index,1]//4),int(size_list[max_scale_index,0]//4)), interpolation=cv2.INTER_NEAREST)
    score_map= (score_map>0).astype('uint8')
    return score_map, variance_map

def skeletonization(pred_center_line, scaling_factor,  max_scale_index):

    score_map_maximum_scale= pred_center_line[max_scale_index]
    score_map= voting(pred_center_line,scaling_factor)
    '''initial predicted segmentation mask'''
    #theta_map= variance_map[2]
    #prepare score map for skeletionization
    score_map= (score_map*255).astype('uint8')


    #score map is a map of 0 and ones.
    #extract contours from it.
    blobs_labels = measure.label(score_map, background=0)
    ids = np.unique(blobs_labels)


    #iterate through score maps and extract a contour
    skeleton_list=[]

    for component_no in range(len(ids)):
        if ids[component_no]==0:
            continue
        current_score_map= (blobs_labels==ids[component_no]).astype('uint8')*255

        # print("current score map")
        # plt.imshow(current_score_map)
        # plt.show()
        # print(np.unique(current_score_map),"current score map")
        contours,hierarchy=cv2.findContours(current_score_map, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        #extract the single contour which is there
        contours= contours[0].squeeze(1)
        #scale up the points
        # print("contours_shape",contours.shape)
        #scale up the points

        #skip if a component has less than 2 points
        if contours.shape[0]<=2:
            continue

        #skeletonize the current score map
        skeleton= skeletonize_image(current_score_map).astype('uint8')
        print("current_score_map",current_score_map.shape)
        #print("current_score_map",np.unique(current_score_map),current_center_line_length)
        #append the skeletons for each component
        skeleton_list.append(skeleton)


    return skeleton_list, score_map

def gaussian_prediction_from_skeleton(component_score_maps,variance_map_x,variance_map_y,theta_map,size_list,max_scale_index):
    #perform gaussian prediction for each component
    sample_contours=[]

    for i,center_line_mask in enumerate(component_score_maps):

        component_variance_map= center_line_mask*(variance_map_x+ variance_map_y)
        component_variance_map_x= center_line_mask*variance_map_x
        component_variance_map_y= center_line_mask*variance_map_y
        component_theta_map= center_line_mask*theta_map
        non_zero_arrays=np.nonzero(component_variance_map)
        n_gaussians= non_zero_arrays[0].shape[0]
        component_gaussians=[]
        #print("n_gaussians",n_gaussians,"for image",image_id)

        if n_gaussians:

            for j in range(n_gaussians):
                x_coordinate=non_zero_arrays[0][j]
                y_coordinate= non_zero_arrays[1][j]
                gaussian=create_gaussian_array(
                                        component_variance_map_x[x_coordinate][y_coordinate],\
                                        component_variance_map_y[x_coordinate][y_coordinate],\
                                        component_theta_map[x_coordinate][y_coordinate],\
                                        x_coordinate,\
                                        y_coordinate,\
                                        height=(size_list[max_scale_index,1]//4),\
                                        width=(size_list[max_scale_index,0]//4)
                                        )

                component_gaussians.append(gaussian)

            component_gaussians= np.stack(component_gaussians,0)
            component_gaussians= np.max(component_gaussians,0)
            component_gaussians= (component_gaussians>0.60).astype(int)
            component_gaussians=component_gaussians.astype('uint8')
            print("component gaussian ki shape",component_gaussians.shape)
            contours,hierarchy=cv2.findContours(component_gaussians, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            filtered_contour_list_by_points=[contours[k].squeeze(1) for k in range(len(contours)) if len(contours[k])>=25]
            filtered_contour_list=[k for k in filtered_contour_list_by_points if area(k[:,0].tolist(),k[:,1].tolist()) ]
            sample_contours+=filtered_contour_list
    print("godzilla mode",len(sample_contours),len(component_score_maps))
    return sample_contours


def writing_predictions(result_dir, filtered_contours,image_id):
    (result_dir/'predictions').mkdir(exist_ok=True, parents=True)
    pred_dump_path= str(result_dir/'predictions')+ '/'+image_id+'.txt'
    #print("pred_dump_path",pred_dump_path)
    print("image id",image_id)
    if os.path.exists(pred_dump_path):
        os.remove(pred_dump_path)
    fid = open(pred_dump_path, 'a')
    content=''
    #writing the actual predictions for Deteval
    for filtered_contours in filtered_contour_list:

        #skip the contours that dont satisfy the area threshold
        if filtered_contours=='':
            continue
        rows,cols= filtered_contours.shape
        for row in range(rows):

            item=filtered_contours[row,:]
            x= item[0]
            y= item[1]
            content+=str(y)+','+str(x)+','
        #remove comm
        content= content[:-1]+'\n'
        fid.write(content)
        #reset the content string
        content=''
    fid.close()


def write_center_line_prediction(skeleton_list, ground_truth_skeleton, segmetation_map, image_path):
    image = cv2.resize(cv2.imread(image_path),(ground_truth_skeleton.shape[1], ground_truth_skeleton.shape[0]))
    test_skeleton= np.zeros(ground_truth_skeleton.shape)
    ''' skeletonization predicted segmentation mask'''

    for skeleton in skeleton_list:
        test_skeleton+=skeleton
    print("combined skeleton",np.unique(test_skeleton))
    print('BUUUUUUUUUUNNNNNNNNNNIYYYYYYYYYYYAAAAAAAA',ground_truth_skeleton.shape,segmentation_map.shape,test_skeleton.shape  )
    image[:,:,0] = image[:,:,0]*test_skeleton
    image[:,:,1] = image[:,:,1]*test_skeleton
    score_map_to_dump=np.concatenate((ground_truth_skeleton*255,segmentation_map,test_skeleton*255),axis =1)
    # plt.imshow(score_map_to_dump)
    # plt.show()
    # plt.imshow(image)
    # plt.show()




def visualize_prediction(filtered_contour_list, image_path, size, width=5):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size[1], size[0]))
    cv2.drawContours(image,filtered_contour_list,-1,(0,255,255),width)
    # plt.imshow(image)
    # plt.show()
    return 0



# main testing code

if __name__ == '__main__':

    if os.path.exists('./output.txt'):
        os.remove('./output.txt')
    #delete the result folder if exists
    paths= ['results/predictions',\
            'results/variance_maps',\
            'results/filteredcontours',\
            'results/center_line',\
            'results/predictions_with_area'
            ]
    for path in paths:
        if os.path.isdir(path):
            sh.rmtree(path)
    #make result directories
    local_data_path = Path('.').absolute()
    result_dir=local_data_path/'results'
    local_data_path.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True, parents=True)

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    #scales= [512,512+128,512+2*128]
    #scales= [512,512+128,512+2*128]
    scaling_factor= [1,0.75,0.65,0.5]
    test_loader= TestDataLoader(max_size=512+2*128, scaling_factor=scaling_factor)
    model=EAST(segmentation_threshold=0.10).to(device)
    checkpoint= 'snapshots/TotalText_3d_rotated_gaussian_attention_30000.pth'
    model.load_state_dict(torch.load(checkpoint,map_location=device),strict=True)
    model.eval()


    for i,batch in enumerate(test_loader):
        print("sample_id: ",i)
        scaled_images,scaled_center_line_images,meta, size_list= batch


        size_list = np.asarray(size_list)
        image_path= meta['image_path']

        #print('path of the image ', image_path,"godzilla",type(meta['image_path']))
        #input('halt')
        image_id= image_path.split('/')[-1].split('.')[0]


        # check the image size and meta deta
        for index in range(len(scaling_factor)):
            print(scaled_images[index].shape, scaled_center_line_images[index].shape)


        H_orig,W_orig= meta['image_shape']
        #print('original image size', H_orig, W_orig)
        max_scale_index = np.argmax(np.asarray(scaling_factor))

        ground_truth_skeleton = scaled_center_line_images[max_scale_index]

        # plt.imshow(ground_truth_skeleton)
        # plt.show()

        pred_center_line= []
        pred_variance_map= []

        for image in scaled_images:
            #print("==================================================image size :",image.shape)
            image=torch.from_numpy(image).unsqueeze(0)
            image= image.to(device)
            score_map, variance_map = model_forward_pass(model, image,max_scale_index)
            pred_center_line.append(score_map)
            pred_variance_map.append(variance_map)

        #extract variance parameters for max scale
        variance_map= pred_variance_map[max_scale_index]
        variance_map_x= variance_map[0]
        variance_map_y= variance_map[1]
        theta_map= 3.14*F.sigmoid(torch.from_numpy(variance_map[2])).numpy()

        skeleton_list, segmentation_map = skeletonization(pred_center_line,scaling_factor, max_scale_index)
        write_center_line_prediction(skeleton_list, ground_truth_skeleton, segmentation_map, image_path)

        sample_contours = gaussian_prediction_from_skeleton(skeleton_list,variance_map_x,variance_map_y,theta_map,size_list,max_scale_index)

        filtered_contour_list=sample_contours
        #visualize_prediction(filtered_contour_list, image_path, [int(size_list[max_scale_index,0]//4),int(size_list[max_scale_index,1]//4)], width=1)
        #print("calculating scaling factor",W_orig, H_orig,size_list[max_scale_index,1],size_list[max_scale_index,0])
        scaling_factor_x= W_orig/(size_list[max_scale_index,1]*0.25)
        scaling_factor_y= H_orig/(size_list[max_scale_index,0]*0.25)
        #print("scaling factor is:",scaling_factor_x,scaling_factor_y)
        #print("len of fultered conoout list",len(filtered_contour_list))
        for j in range(len(filtered_contour_list)):
            scaling=[scaling_factor_x,scaling_factor_y]
            scaling=np.array(scaling).T
            #print("dtype",filtered_contour_list[0].dtype,type(filtered_contour_list[j]))
            if filtered_contour_list[j]=='':
                continue
            #print("filtered contour list shape",filtered_contour_list[j].shape)
            #print("scaling cases",scaling)
            filtered_contour_list[j]=filtered_contour_list[j]*scaling
            filtered_contour_list[j] = filtered_contour_list[j].astype(int)

        visualize_prediction(filtered_contour_list, image_path, [H_orig, W_orig])
        writing_predictions(result_dir, filtered_contour_list,image_id)
