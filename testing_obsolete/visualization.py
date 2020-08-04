import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from pathlib import Path
import argparse
import scipy.io as io


local_data_path = Path('.').absolute()
parser = argparse.ArgumentParser(description="Welcome to the Segression Testing Module!!!")

parser.add_argument("--dataset", type=str, default="TOTALTEXT",
                    help="Dataset on which testing is to be done, (TOTALTEXT,CTW1500,MSRATD500,ICDAR2015)")

parser.add_argument("--pred-dir", type=str, default="results/TOTALTEXTgaussian_threshold=0.6_segmentation_threshold=0.4",
                    help="Path to the snapshot to be used for testing")
parser.add_argument("--dir-root", type=str, default="",
                    help="Enter the dir from where gt is to be picked up")

# parser.add_argument("--dump-dir", type=str, default="results/TOTALTEXTgaussian_threshold=0.6_segmentation_threshold=0.4",
#                     help="Path to the snapshot to be used for testing")



args = parser.parse_args()

dump_dir= args.pred_dir+'_preds_dump'
(local_data_path/dump_dir).mkdir(exist_ok=True, parents=True)


def load_dataset_images(path):
    dataset=[]
    image_ext=['jpg','JPG','png']
    for ext in image_ext:
        for file in glob.glob(os.path.join(path, "*."+ext)):
            #print(file)
            dataset.append(file)
    #print(dataset)
    return dataset

def translation(self,delta_x, delta_y):
    translation_matrix = np.asarray([[1,0,delta_x],[0,1,delta_y],[0,0,1]])
    return translation_matrix

def rotation(self, theta):
    rotation_matrix = np.asarray([[math.cos(theta), -math.sin(theta),0],[math.sin(theta), math.cos(theta),0],[0,0,1]])
    return rotation_matrix

def convert_to_rotated_bounding_box(self,x_0, y_0, w, h, theta):
    '''
    arguments:
        (x_0, y_0) = top left coordinate of the horizontal reactangle
        w = width of the rectangle
        h = height of the reactangle
        theate = angle in radian with reference to the center of the rectangle
    return :
        bounding_box = boudning box with (x,y): 4 x 2
    '''
    coordinate = np.asarray([[x_0, y_0,1],[x_0+w, y_0,1],[x_0+w, y_0 + h,1],[x_0,y_0+h,1]]).T
    center_coordinate_x = (x_0 +  w//2)
    center_coordinate_y = (y_0 +  h//2)
    transformation_matrix = np.matmul(self. translation(center_coordinate_x, \
                            center_coordinate_y),np.matmul(self.rotation(theta), \
                            self.translation(-center_coordinate_x, -center_coordinate_y)))
    rotated_coordinate = np.matmul(transformation_matrix, coordinate)
    bounding_box = rotated_coordinate.T

    bounding_box = bounding_box[:,0:2]
    return bounding_box


def parse_ground_truth(filename, dataset):
    print('image_path')
    polygons = []
    if dataset=='TOTALTEXT':
        print("total text claled")
        #print("true",mat_path)
        gt_path=os.path.join(gt_dir,'poly_gt_'+filename+'.mat')
        annot = io.loadmat(gt_path)
        #print("before annotations",annot)
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            if text!='#': 
                pts = np.stack([x, y]).T.astype(np.int32)
                polygons.append(pts)
    elif dataset=='CTW1500':
        gt_path=os.path.join(gt_dir,filename+'.txt')
        with open(gt_path) as f:
            lines= [line.strip().split(',') for line in f.readlines()]
            for line in lines:
                line = [int(i) for i in line]
                x= []
                y= []
                offset=4
                for i in range(7):
                    x.append(line[offset+2*i]+line[0])
                    y.append(line[offset+2*i+1]+line[1])
                offset=9
                for i in range(7):
                    #print(2*(offset+i)+1)
                    x.append(line[2*(offset+i)]+line[0])
                    y.append(line[2*(offset+i)+1]+line[1])
                    #print("i is",i,"x old",line[offset+2*i],"x after",line[offset+2*i]+line[0])
                pts=np.stack([x, y]).T.astype(np.int32)
                text= 'dummy'
                ori= 'c'
                polygons.append(pts)

    elif dataset=='MSRATD500':
        gt_path=os.path.join(gt_dir,filename+'.txt')
        with open(gt_path) as f:
            lines= [line.strip().split(',') for line in f.readlines()]

        for line in lines:
            #print("line is",line)
            line = line[0].split(' ')
            x_0= int(line[2])
            y_0= int(line[3])
            w= int(line[4])
            h= int(line[5])
            theta= float(line[6])

            #print('theta ====================================',theta)

            bounding_box = self.convert_to_rotated_bounding_box(x_0, y_0, w, h, theta)
            x = bounding_box[:,0]
            y = bounding_box[:,1]
            pts=np.stack([x, y]).T.astype(np.int32)
            text= 'dummy'
            ori= 'c'
            polygons.append(pts)

    elif dataset=='ICDAR2015':
        gt_path=os.path.join(gt_dir,'gt_'+filename+'.txt')
        with open(gt_path) as f:
            lines= f.readlines()
        for line in lines:
            temp = line.rstrip('\n').lstrip('\ufeff').split(',')
            annotation = temp[-1]
            #print(temp)
            bounding_box =[]
            for index in range(8):
                bounding_box.append(int(temp[index]))
            bounding_box = np.asarray(bounding_box)
            bounding_box = bounding_box.reshape(4,2)

            x = bounding_box[:,0]
            y = bounding_box[:,1]
            pts=np.stack([x, y]).T.astype(np.int32)

            ori= 'c'
            if annotation == '###':
                text= '#'
            else:
                text= 'dummy'
            polygons.append(pts)
    else:
        raise Exception("Invalid Dataset given")

    return polygons


def parse_prediction_file(filename):
    predictions=[]
    prediction_file_path = os.path.join(args.pred_dir,'res_'+filename+'.txt' )
    f= open(prediction_file_path, 'r')
    lines= f.readlines()
    f.close()
    for line in lines:
        #print(line)
        line = line.split(',')
        coord=[]
        for index in range(len(line)):
            coord.append(int(line[index]))
        coord = np.asarray(coord)
        coord=coord.reshape(-1,2)
        # if args.dataset=='TOTALTEXT':
        #     coord_swap=np.zeros(coord.shape)
        #     coord_swap[:,0]=coord[:,1]
        #     coord_swap[:,1]=coord[:,0]
        #     coord_swap=coord_swap.astype(int)
        #     #print(coord.shape, coord_swap.shape)
        #     predictions.append(coord_swap)
        # else:
        #     predictions.append(coord)
        predictions.append(coord)
    return predictions

def dump_visualization(image_path, gt, predictions):
    original_image = cv2.imread(image_path)
    ground_truth_image = original_image.copy()
    prediction_image = original_image.copy()


    for index in range(len(gt)):
        pts = gt[index]
        cv2.drawContours(ground_truth_image, [pts],0,(255,0,255), 5)
    for index in range(len(predictions)):
        pts=predictions[index]
        #print("sample visualization",index, " out of ",len(predictions))
        #print(pts.shape, pts)
        cv2.drawContours(prediction_image, [pts],0,(255,0,255), 5)
    im_to_save= np.concatenate((original_image,\
    ground_truth_image,prediction_image),axis=1)
    # plt.imshow(im_to_save)
    # plt.show()
    cv2.imwrite(os.path.join(dump_dir,image_path.split(os.sep)[-1]),im_to_save)
    return 0

if args.dataset=='CTW1500':
    dir_root = 'data/ctw-1500'
elif args.dataset=='MSRATD500':
    dir_root= 'data/msra-td500'
elif args.dataset=='ICDAR2015':
    dir_root = 'data/icdar-2015'
elif args.dataset=='TOTALTEXT':
    dir_root= 'data/total-text'
else:
    raise Exception("Invalid Dataset given")
#manually set the path if passed as argument 
if args.dir_root:
    dir_root= args.dir_root
#prediction_path='results/TOTALTEXTgaussian_threshold=0.6_segmentation_threshold=0.4'
image_dir = os.path.join(dir_root, 'Images/Test')
gt_dir =os.path.join(dir_root, 'gt/Test')
dataset= load_dataset_images(image_dir)
for i,image_file_name in enumerate(dataset):
    print("image file name",image_file_name)
    filename = image_file_name.split(os.sep)[-1].split('.')[0]
    print("fffff",filename)
    ground_truth_annotation = parse_ground_truth(filename,\
                              dataset=args.dataset)
    #print(ground_truth_annotation)
    prediction_annotaiton = parse_prediction_file(filename)
    print("processing",i, " out of",len(dataset), "samples")
    dump_visualization(image_file_name,\
    ground_truth_annotation, prediction_annotaiton)
