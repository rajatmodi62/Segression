import scipy.io as io
import numpy as np
import os

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class TotalText(TextDataset):

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
        #print("before filteringimage list",self.image_list[0:5])
        #remove the ignore images from the list
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))

        #print("after filtering image list",self.image_list[0:5])
        self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        #constructed a list containing the dierctory and annottationsin the mat format
    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        #print("matrix parser called")
        #print("rajat",mat_path.split('.')[-1])
        #if mat_path.split('.')[-1]== 'mat':
        if mat_path.split('.')[-1]=='mat':
            #print("true",mat_path)
            annot = io.loadmat(mat_path)
            #print("before annotations",annot)
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
            #print("type",type(annot['polygt']))
        else:
            #print("in else",mat_path)
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

    def __getitem__(self, item):

        image_id = self.image_list[item]
        #print("image_id",image_id)
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        #print("annotation_id",annotation_id)
        #correct the annotation_id for txt case of chinese dataset
        if annotation_id.split('.')[0].split('_')[-1]=='synth':
            annotation_id= annotation_id.split('.')[0]
            splits= annotation_id.split('_')
            annotation_id= ''
            for split in splits[2:]:
                annotation_id+= split+'_'
            annotation_id= annotation_id[:-1]+'.txt'
            #print("filtered annotation id ",annotation_id)

            #print("corrected_annotation_id",annotation_id)
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)
        #print("godzilla",annotation_path)
        #print the image id
        #print("image id",image_id)
        #print("polygons",polygons)
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

    trainset = TotalText(
        data_root='data/total-text',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    for idx in range(0, 10):
        image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta,_ = trainset[idx]
        print("done!!")
        #print(type(img),img.shape,train_mask.shape,center_line_map.shape)
