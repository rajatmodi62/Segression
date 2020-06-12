import cv2
import numpy as np
import matplotlib.pyplot as plt
#
# def skeletonize(img,img_path='test.jpg'):
#
#     #img = cv2.imread(img_path,0)
#     print("function called",img.shape,np.unique(img))
#     size = np.size(img)
#     skel = np.zeros(img.shape,np.uint8)
#     ret,img = cv2.threshold(img,127,255,0)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#     done = False
#
#     while( not done):
#         eroded = cv2.erode(img,element)
#         temp = cv2.dilate(eroded,element)
#         temp = cv2.subtract(img,temp)
#         skel = cv2.bitwise_or(skel,temp)
#         img = eroded.copy()
#
#         zeros = size - cv2.countNonZero(img)
#         if zeros==size:
#             done = True
#     return skel


from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

def skeletonize_image(img,img_path='test.jpg'):
    # plt.imshow(img)
    # plt.show()
    img = (img>125)*1
    #ret,img = cv2.threshold(img,50,255,0)
    #print(np.unique(img))
    #img=(img==255).astype('uint8')
    #plt.imshow(img)
    #plt.show()
    skeleton = skeletonize(img)
    # plt.imshow(skeleton*255)
    # plt.show()
    return skeleton
