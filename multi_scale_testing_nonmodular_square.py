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
#set gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load model
model=EAST(segmentation_threshold=0.10).to(device)
#load checkpoint
checkpoint= 'snapshots/TotalText_3d_rotated_gaussian_attention_30000.pth'
model.load_state_dict(torch.load(checkpoint,map_location=device),strict=True)
model.eval()

#input images have to be normalized for imagenet weights
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)

#initialize testset
testset= TestDataLoader(scales=scales)
#construct dataloader
test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)



#enumerate over the DataLoader
for i,batch in enumerate(test_loader):

    print("sample_id: ",i)
    scaled_images,scaled_center_line_images,meta= batch
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
        score_map,contour_map, flag,variance_map=model(image)

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

        #print("score map",score_map.shape)
        #print("contour map",contour_map.shape)
        #print("variance feature map",variance_map.shape)

    #perform voting here
    score_map_maximum_scale= pred_center_line[-1]
    score_map= voting(pred_center_line)
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

    scaled_center_line_images=scaled_center_line_images[-1].squeeze(0).numpy()

    score_map_to_dump=np.concatenate((scaled_center_line_images*255,score_map_before,score_map_after*255),axis =1)
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

    image_path= meta['image_path'][0]
    #print("in eval",type(image_path),image_path)
    #extract image_id
    image_id= image_path.split('/')[-1].split('.')[0]
    #make the pred dump path
    (result_dir/'predictions').mkdir(exist_ok=True, parents=True)
    pred_dump_path= str(result_dir/'predictions')+ '/'+image_id+'.txt'
    #print("pred_dump_path",pred_dump_path)

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

    #dump the voted center line
    (result_dir/'center_line').mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(result_dir/'center_line'/(image_id+'.jpg')),score_map_to_dump)
    (result_dir/'variance_maps').mkdir(exist_ok=True, parents=True)
    #print("max",np.max(variance_map_x),np.max(theta_map))
    variance_map= np.concatenate((variance_map_x/np.max(variance_map_x),variance_map_y/np.max(variance_map_y),theta_map/np.max(theta_map)),axis=1)
    plt.imsave(str(result_dir/'variance_maps'/(image_id+'.jpg')),variance_map)
    # plt.imshow(theta_map)
    # plt.show()
    #cv2.imwrite(str(result_dir/'variance_maps'/(image_id+'.jpg')),variance_map*255)

    ''' dump the area predictions'''
    #dump the area directory
    (result_dir/'predictions_with_area').mkdir(exist_ok=True, parents=True)

    pred_dump_path= str(result_dir/'predictions_with_area')+ '/'+image_id+'.txt'
    #print("pred_dump_path",pred_dump_path)

    if os.path.exists(pred_dump_path):
        os.remove(pred_dump_path)
    fid = open(pred_dump_path, 'a')
    content=''
    #writing the actual predictions for Deteval

    for i,filtered_contours in enumerate(filtered_contour_list):
        content= "area="+ str(area_list[i])+ ',' + "center_line_length=" +str(center_line_length_list[i])+ ','
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
