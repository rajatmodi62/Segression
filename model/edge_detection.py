import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


#cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EdgeDetection(nn.Module):

    def __init__(self):
        super(EdgeDetection, self).__init__()
        print("edge detection")
        self.sobel_kernel= torch.from_numpy(np.array([[1., 0., -1.],
                                     [2., 0., -2.],
                                     [1., 0., -1.]]))
        self.sobel_kernel_horizontal=self.sobel_kernel.unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_horizontal=self.sobel_kernel_horizontal.type(torch.FloatTensor).to(device)

        self.sobel_kernel_vertical = self.sobel_kernel.T.unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_vertical= self.sobel_kernel_vertical.type(torch.FloatTensor).to(device)



    def forward(self,segmentation_map,threshold=0.3):
        #segmentation_map = segmentation_map.unsqueeze(1)#.unsqueeze(0)
        horizontal_grad = F.conv2d(segmentation_map, self.sobel_kernel_horizontal,padding=self.sobel_kernel.shape[0]//2)
        vertical_grad = F.conv2d(segmentation_map, self.sobel_kernel_vertical, padding=self.sobel_kernel.shape[0]//2)
        magnitude = torch.sqrt(horizontal_grad**2 + vertical_grad**2)
        #print("before calling",magnitude.shape,magnitude)
        #max_response = torch.max(magnitude.squeeze())
        #print("===================>max resonse",max_response)
        #magnitude = magnitude/max(max_response,1)
        #print("unique values",torch.max(magnitude), torch.min(magnitude))
        # magnitude= torch.gt(magnitude,threshold)*1.0
        magnitude= torch.gt(magnitude,threshold)*(segmentation_map.squeeze())
        return magnitude


if __name__ == '__main__':
    ed = EdgeDetection()
    # import cv2
    # mona_lisa= cv2.imread('monalisa.jpg',0)
    # plt.imshow(mona_lisa)
    # plt.show()
    # seg_map= torch.from_numpy(mona_lisa).unsqueeze(0).unsqueeze(0)
    # seg_map = seg_map.type(torch.FloatTensor)
    seg_map = torch.zeros(2,1,250,250)
    seg_map[0,:,50:100,50:100] = 1
    seg_map[1,:,10:200,10:200] = 1
    mag= ed(seg_map)
    mag = mag/torch.max(mag)
    mag = torch.gt(mag,0.5)*1.0

    #print('unique value', torch.unique(mag))
    #print("Edge Detection Module", seg_map.shape)
    plt.subplot(1,2,1)
    plt.imshow(seg_map[0].squeeze().numpy())
    plt.subplot(1,2,2)
    plt.imshow(mag[0].squeeze().numpy())
    plt.show()
