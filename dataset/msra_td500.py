import scipy.io as io
import numpy as np
import os
import cv2
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import matplotlib.pyplot as plt
import math


class MSRATD500(TextDataset):

    def __init__(self, data_root,input_size=512,ignore_list=None, is_training=True, transform=None):
        super().__init__(transform)
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
        self.image_list = os.listdir(self.image_root)
        #print("before filteringimage list",self.image_list[0:5],len(self.image_list))
        #remove the ignore images from the list
        self.image_list = list(filter(lambda img: img.replace('.JPG', '') not in ignore_list, self.image_list))

        #print("after filtering image list",len(self.image_list))
        self.annotation_list = ['{}.gt'.format(img_name.replace('.JPG', '')) for img_name in self.image_list]
        #constructed a list containing the dierctory and annottationsin the mat format
        #print("annotation_list",self.annotation_list[:5])

    def translation(self,delta_x, delta_y):
        translation_matrix = np.asarray([[1,0,delta_x],[0,1,delta_y],[0,0,1]])
        return translation_matrix

    def rotation(self, theta):
        rotation_matrix = np.asarray([[math.cos(theta), -math.sin(theta),0],[math.sin(theta), math.cos(theta),0],[0,0,1]])
        #print('rotation matrix ====>', rotation_matrix)
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
        #transformation_matrix = self.translation(-center_coordinate_x, -center_coordinate_y)



        #transformation_matrix = np.asarray([[1,0,0],[0,1,0],[0,0,1]])

        #print(transformation_matrix.shape, coordinate.shape)
        rotated_coordinate = np.matmul(transformation_matrix, coordinate)
        bounding_box = rotated_coordinate.T
        #print('bounding box regression', bounding_box.shape)

        bounding_box = bounding_box[:,0:2]
        #print('bounding box regression==============>', bounding_box)
        return bounding_box


    def parse_txt(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        #print("matrix parser called",mat_path)
        #print('text file name ', mat_path)
        polygons = []
        contours=[]
        #print("matrix parser called",mat_path)
        if mat_path.split('.')[0].split('_')[-1]=='synth':
            #print("===========>IT'S SYNTH GENTLEMEN, IT's SYNTH :-).........")
            with open(mat_path) as f:
                lines = [line.strip().split(',')[0:8] for line in f.readlines()]
                polygons = []
                for line in lines:
                    x=[line[0],line[2],line[4],line[6]]
                    y=[line[1],line[3],line[5],line[7]]
                    # print('======>X',x)
                    # print('======>Y',y)
                    pts = np.stack([x, y]).T.astype(np.int32)
                    text= 'dummy'
                    ori= 'c'
                    polygons.append(TextInstance(pts, ori, text))
                    contours.append(pts)
                image_id= mat_path.split('/')[-1].split('.')[0]+'.jpg'

        else:
            #print("===========>IT'S BHUNIYA GENTLEMEN, IT's BHUNIYA :-).........")
            with open(mat_path) as f:
                lines= [line.strip().split(',') for line in f.readlines()]

            image_id= mat_path.split('/')[-1].split('.')[0]+'.JPG'
            #print("================>BHUNIYA IMAGE ID ",image_id)


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
                polygons.append(TextInstance(pts, ori, text))
                contours.append(pts)


        image_dir= 'data/msra-td500/Images/Train/' + image_id
        #print("image_dir",image_dir)
        # image = pil_load_img(image_dir)
        #print("length od contours",len(contours))
        # image= cv2.drawContours(image,contours,-1,(0,255,255),5)
        # plt.imshow(image)
        # plt.show()
        return polygons


    def __getitem__(self, item):

        image_id = self.image_list[item]
        #print("image_id",image_id)
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        #print("getitem image shape is",image.shape)
        # Read annotation
        annotation_id = self.annotation_list[item]
        #print("annotation_id",annotation_id)
        #correct the annotation_id for txt case of chinese dataset
        if annotation_id.split('.')[0].split('_')[-1]=='synth':
            #print("annotation_id",annotation_id)
            annotation_id= annotation_id.split('.')[0]
            annotation_id+= '.txt'
            #
            # splits= annotation_id.split('_')
            # annotation_id= ''
            # for split in splits[2:]:
            #     annotation_id+= split+'_'
            # annotation_id= annotation_id[:-1]+'.txt'
            #print("filtered annotation id ",annotation_id)

            #print("corrected_annotation_id",annotation_id)
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_txt(annotation_path)
        #print("godzilla",annotation_path)
        #print the image id
        #print("image id",image_id)
        # print("polygons",polygons)
        for i, polygon in enumerate(polygons):
            #print("now polygon is",polygon,type(polygon))
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()
        '''returning the ground truth .mat for detEval'''
        #print("dataloader image path",image_path)
        #print("godzilla")
        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path,gt_mat_path=annotation_path)

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

    trainset = MSRATD500(
        data_root='data/msra-td500',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    print("lne of trainset",len(trainset))
    for idx in range(0, len(trainset)):
        print("doing!!",idx)

        image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = trainset[idx]
        print("done")
        #print(type(img),img.shape,train_mask.shape,center_line_map.shape)
#240,371,488
