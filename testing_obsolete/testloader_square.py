import cv2
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util.augmentation import BaseTransform

def pil_load_img(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image)
    return image

'''
Input: Dataset which is to be tested
'''


class TestDataLoader(data.Dataset):

    def __init__(self,dataset='TOTALTEXT',scales=[512],test_dir=""):
        super().__init__()
        #get the list of image path

        if dataset=='CTW1500':

            self.test_img_dir= 'data/ctw-1500/Images/Test'

        elif dataset=='MSRATD500':

            self.test_img_dir= 'data/msra-td500/Images/Test'

        elif dataset=='ICDAR2015':

            self.test_img_dir= 'data/icdar-2015/Images/Test'

        elif dataset=='TOTALTEXT':

            self.test_img_dir= 'data/total-text/Images/Test'

        else:
            raise Exception("Invalid Dataset given")
        
        #update the testing dir to the passed test_dir otherwise 
        if test_dir:
            self.test_img_dir=test_dir
        self.test_img_path= [os.path.join(self.test_img_dir,path) \
                            for path in os.listdir(self.test_img_dir)]

        self.means = (0.485, 0.456, 0.406)
        self.stds = (0.229, 0.224, 0.225)
        self.scales= scales

        print("total images in ",dataset, ":",len(self.test_img_path))

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, idx):
        image = pil_load_img(self.test_img_path[idx])
        img_path=self.test_img_path[idx]
        img_id= img_path.split('/')[-1]
        #print("image",image.shape)
        H, W, _ = image.shape
        scaled_images= []
        # print("fetching image")
        # plt.imshow(image)
        # plt.show()
        for scale in self.scales:
            #create a transform here
            transform = BaseTransform(
                size=(scale,scale), mean=self.means, std=self.stds
            )

            scaled_images.append(transform(image)[0].transpose(2,0,1))

        #print("in dataloader",image_path)
        meta = {
            'image_id': img_id,
            'image_path': img_path,
            'image_shape' : (H,W)
        }
        return scaled_images,meta
if __name__ == '__main__':

    print("main!!")

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    scales= [512,512+128,512+2*128]
    print("calling transform")
    transform = BaseTransform(
        size=512, mean=means, std=stds
    )
    print("calling test dataloader")
    testset= TestDataLoader(dataset='CTW1500',scales=scales)
    for i in range(len(testset)):
        testset[i]
        print(i)
