#change dataloader shuffle to false
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
from model.segression import Segression
from util.augmentation import BaseTransform
from shapely.geometry import Polygon
from skeletonize import skeletonize_image
import shutil as sh
from pathlib import Path
import math
import torch.nn.functional as F
import argparse

print("All requisite testing modules loaded")

#free the gpus
os.system("nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9")
#enter scales in a sorted increasing order
scales= [512-128,512,512+128,512+2*128]
#256,512,512+256,512+128
# scales= [512+2*128]

BATCH_SIZE = 1
INPUT_SIZE = 256

def area(x, y):
    polygon = Polygon(np.stack([x, y], axis=1))
    #print("shape of stack",np.stack([x, y], axis=1).shape)
    return float(polygon.area)

def create_gaussian_array(variance_x,variance_y,theta,x,y,height=INPUT_SIZE//4,width=INPUT_SIZE//4):
    # i varies by height
    #print("shapes",variance_x.shape,variance_y.shape,theta.shape)
    sin = math.sin(theta)
    cos = math.cos(theta)
    var_x = math.pow(variance_x,2)
    var_y = math.pow(variance_y,2)
    a= (cos*cos)/(2*var_x) + (sin*sin)/(2*var_y)
    b= (-2*sin*cos)/(4*var_x) + (2*sin*cos)/(4*var_y)
    c= (sin*sin)/(2*var_x) + (cos*cos)/(2*var_y)
    #
    i= np.array(np.linspace(0,height-1,height))
    i= np.expand_dims(i,axis=1)
    i=np.repeat(i,height,axis=1)
    j=i.T
    #print("a",a,b,c )
    #the x & y around which gaussian is centered
    A= np.power(i-x,2)
    # B=1
    B= 2*(i-x)*(j-y)
    C= np.power(j-y,2)
    gaussian_array= np.exp(-(a*A+b*B+c*C))
    return gaussian_array

def voting(center_line_list):
    #get the minimum no of votes needed
    min_votes_needed= len(scales)//2 + 1
    #perform the sum in the list
    mask = np.zeros(center_line_list[0].shape)

    for scaled_center_line_image in center_line_list:
        mask+= scaled_center_line_image

    #perform voting
    mask= (mask>= min_votes_needed).astype('uint8')
    return mask


######################### TESTING CODE BEGINS HERE ############################

##################### PARSER ##################################################

parser = argparse.ArgumentParser(description="Welcome to the Segression Testing Module!!!")

parser.add_argument("--dataset", type=str, default="TOTALTEXT",
                    help="Dataset on which testing is to be done, (TOTALTEXT,CTW1500,MSRATD500,ICDAR2015)")
parser.add_argument("--snapshot-dir", type=str, default="snapshots/SynthText_3d_rotated_gaussian_without_attention_200000.pth",
                    help="Path to the snapshot to be used for testing")
parser.add_argument("--segmentation_threshold", type=float, default=0.4,
                    help="Thresholding parameter for predicted center line mask, range (0,1)")
parser.add_argument("--gaussian_threshold", type=float, default=0.6,
                    help="Thresholding parameter for predicted gaussian map, range (0,1)")
parser.add_argument("--backbone", type=str, default="VGG",
                    help="Enter the Backbone of the model, (VGG,RESNEST)")

args = parser.parse_args()

################################################################################
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
#print parser arguments

print("-------------> Dataset:",args.dataset)
print('-------------> snapshot:',args.snapshot_dir)
print('-------------> segmentation threshold:',args.segmentation_threshold)
print('-------------> gaussian threshold:',args.gaussian_threshold)
print('-------------> backbone :',args.backbone)



################################################################################

#make result directories
result_dir=local_data_path/args.dataset/'results'
result_dir.mkdir(exist_ok=True, parents=True)
#set gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load model
model= Segression(center_line_segmentation_threshold=args.segmentation_threshold,\
                    backbone=args.backbone,\
                    segression_dimension= 3,\
                    mode='test').to(device)

#load checkpoint
print("trying to load snapshot: ", args.snapshot_dir)
model.load_state_dict(torch.load(args.snapshot_dir,map_location=device),strict=True)
print("snapshot loaded!!!!")
model.eval()

#input images have to be normalized for imagenet weights
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)

# #initialize testset
# dataset= 'CTW1500'
# # dataset='TOTALTEXT'
# # dataset='MSRATD500'
# # dataset='ICDAR2015'

testset= TestDataLoader(dataset=args.dataset,scales=scales)
#construct dataloader
test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

eval= EvalOutputs(args)


#enumerate over the DataLoader
for i,batch in enumerate(test_loader):

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
        image_scale= image.size()[2]
        image= image.to(device)
        #forward pass

        score_map, variance_map= model(image)
        # if flag==True:
        #     # print('EXCEPTION ------------------------------------>')
        #     continue

        #score map is center line map
        score_map= score_map.squeeze(0)
        score_map= score_map.squeeze(0)
        score_map= score_map.detach().cpu().numpy()
        #contour map is gaussian map
        # contour_map= contour_map.squeeze(0)
        # contour_map= contour_map.squeeze(0)
        # contour_map= contour_map.detach().cpu().numpy()

        #variance map contains the variances of the gaussians
        variance_map= variance_map.squeeze(0)
        variance_map=variance_map.squeeze(0)
        #print("theta shape",variance_map.size())
        theta_map=  3.14*F.sigmoid(variance_map[2,:,:])
        theta_map= theta_map.detach().cpu().numpy()
        #print("theta map p",theta_map.shape)
        variance_map= variance_map.detach().cpu().numpy()

        #upsampling all the images to same scale, max scale is last element
        #note: have to do //4 since the model outputs the images of //4 size.
        max_scale= scales[-1]//4

        #need to add threshold for performing voting
        score_map= (score_map>0.40).astype('uint8')
        score_map=cv2.resize(score_map,(max_scale,max_scale), interpolation=cv2.INTER_NEAREST)

        score_map= (score_map>0).astype('uint8')

        #append the center line map and variance map for each scale
        pred_center_line.append(score_map)
        pred_variance_map.append(variance_map)
        del score_map,variance_map
        #print("score map",score_map.shape)
        #print("contour map",contour_map.shape)
        #print("variance feature map",variance_map.shape)

    #perform voting here
    score_map_maximum_scale= pred_center_line[-1]
    score_map= voting(pred_center_line)
    # plt.imshow(score_map)
    # plt.show()
    #   score_map= ((score_map+ score_map_maximum_scale)>0)*1
    #print("after voting",np.unique(score_map),np.sum((score_map==1).astype('uint8')),score_map.shape)

    #the variance map to draw the gaussian will be the variance of maximum scale
    variance_map= pred_variance_map[-1]
    #print("variance map shape", variance_map.shape)
    variance_map_x= variance_map[0]
    variance_map_y= variance_map[1]
    #theta_map= variance_map[2]
    #prepare score map for skeletionization
    score_map_before= score_map*255
    score_map= (score_map*255).astype('uint8')


    #score map is a map of 0 and ones.
    #extract contours from it.
    blobs_labels = measure.label(score_map, background=0)
    ids = np.unique(blobs_labels)

    #iterate through score maps and extract a contour
    component_score_maps=[]
    component_areas= []
    component_center_line_length= []
    for component_no in range(len(ids)):
        if ids[component_no]==0:
            continue
        current_score_map= (blobs_labels==ids[component_no]).astype('uint8')*255
        # print(np.unique(current_score_map),"current score map")
        contours,hierarchy=cv2.findContours(current_score_map, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        #extract the single contour which is there
        contours= contours[0].squeeze(1)
        #scale up the points
        # print("contours_shape",contours.shape)
        #scale up the points
        scaling=[scaling_factor_x,scaling_factor_y]
        scaling=np.array(scaling).T
        contours=contours*scaling
        #skip if a component has less than 2 points
        if contours.shape[0]<=2:
            continue
        #find the area of these contours
        current_area= area(contours[:,0].tolist(), contours[:,1].tolist())
        # print("current_Area",current_area)

        #skeletonize the current score map
        current_score_map= skeletonize_image(current_score_map).astype('uint8')
        print("current_score_map",current_score_map.shape)
        modified_score_map= current_score_map.copy()
        #scale up to maximumscale/4
        modified_score_map=cv2.resize(modified_score_map,((512+2*128)//4,(512+2*128)//4), interpolation=cv2.INTER_NEAREST)
        modified_score_map= (modified_score_map>0)*1.0
        current_center_line_length= np.sum(modified_score_map)

        #print("current_score_map",np.unique(current_score_map),current_center_line_length)
        #append the skeletons for each component
        component_score_maps.append(current_score_map)
        #append the areas of each component

        component_areas.append(current_area)
        component_center_line_length.append(current_center_line_length)
    #print("finished")

    #construct the score map after
    score_map_after= np.zeros(score_map.shape)
    for map in component_score_maps:
        score_map_after+= map
    score_map_after= (score_map_after>0).astype('uint8')

    #scaled_center_line_images=scaled_center_line_images[-1].squeeze(0).numpy()

    #score_map_to_dump=np.concatenate((scaled_center_line_images*255,score_map_before,score_map_after*255),axis =1)
    image_path= meta['image_path'][0]
    image_id= image_path.split('/')[-1].split('.')[0]


    #perform gaussian prediction for each component
    visual_image = np.zeros((score_map[0].shape),dtype='uint8')
    sample_contours=[]
    area_list= []
    center_line_length_list=[]
    print("len of list",len(sample_contours))
    for i,center_line_mask in enumerate(component_score_maps):
        segmentation_area= component_areas[i]
        center_line_length= component_center_line_length[i]
        component_variance_map= center_line_mask*(variance_map_x+ variance_map_y)
        #print("full variance map")
        component_variance_map_x= center_line_mask*variance_map_x
        component_variance_map_y= center_line_mask*variance_map_y
        component_theta_map= center_line_mask*theta_map
        non_zero_arrays=np.nonzero(component_variance_map)
        n_gaussians= non_zero_arrays[0].shape[0]
        print("i",i)
        component_gaussians=[]
        #print("n_gaussians",n_gaussians,"for image",image_id)
        if n_gaussians:
            #print("entering")
            for j in range(n_gaussians):
                x_coordinate=non_zero_arrays[0][j]
                y_coordinate= non_zero_arrays[1][j]
                #print("during gaussian",component_variance_map.shape,scales[-1]//4)
                #print("gaussian variance is ",component_variance_map[x_coordinate][y_coordinate])
                gaussian=create_gaussian_array(component_variance_map_x[x_coordinate][y_coordinate],component_variance_map_y[x_coordinate][y_coordinate],component_theta_map[x_coordinate][y_coordinate],x_coordinate,y_coordinate,height=(scales[-1]//4),width=(scales[-1]//4))
                component_gaussians.append(gaussian)
                if j%100 is 0:
                    print(j)
            component_gaussians= np.stack(component_gaussians,0)
            component_gaussians= np.max(component_gaussians,0)
            component_gaussians= (component_gaussians>0.60).astype(int)
            component_gaussians=component_gaussians.astype('uint8')

            #contours,hierarchy=cv2.findContours(component_gaussians, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

            contours,hierarchy=cv2.findContours(component_gaussians, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(visual_image, contours, 0, (255, 255, 255), 1)
            # print("visual image")
            # plt.imshow(visual_image)
            # plt.show()
            filtered_contour_list_by_points=[contours[k].squeeze(1) for k in range(len(contours)) if len(contours[k])>=25]
            filtered_contour_list=[k for k in filtered_contour_list_by_points if area(k[:,0].tolist(),k[:,1].tolist()) ]
            #print("updaing sample contours",len(filtered_contour_list))
            if len(filtered_contour_list)==0:
                sample_contours.append('')
                area_list.append(segmentation_area)
                center_line_length_list.append(center_line_length)
            else:
                for filtered_list in filtered_contour_list:
                    sample_contours.append(filtered_list)
                    area_list.append(segmentation_area)
                    center_line_length_list.append(center_line_length)

        else:
            sample_contours+=['']
            area_list.append(segmentation_area)
            center_line_length_list.append(center_line_length)

        #print("center line mask shape",center_line_mask.shape)
        #print("variance map shape",variance_map.shape)
    #print("godzilla")
    print(len(sample_contours),len(component_score_maps),len(area_list))

    filtered_contour_list=sample_contours

    for j in range(len(filtered_contour_list)):

        scaling=[scaling_factor_x,scaling_factor_y]
        scaling=np.array(scaling).T
        #print("dtype",filtered_contour_list[0].dtype,type(filtered_contour_list[j]))
        if filtered_contour_list[j]=='':
            continue
        filtered_contour_list[j]=filtered_contour_list[j]*scaling
        filtered_contour_list[j] = filtered_contour_list[j].astype(int)

    #print(" filtered contour shape",filtered_contour_list[0].shape)
    print(" meta",meta["image_id"])

    #create dataset eval
    print("calling evaluation code upon the dataset")
    eval.generate_predictions(filtered_contour_list,meta["image_id"][0])
