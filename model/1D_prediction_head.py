import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


'''
Prediction Head takes a segmentation map as input.
Variance Map for segmented pixels is used to create a gaussian.
Gaussians cascaded, max pooled & thresholded.

Input:
Variance Map: [N,1,H,W]
Segmentation Map: [N,1,H,W] tensor of 0/1's

Parameters:
Gaussian Kernel containing a single variance.

Returns:
[1,1,H,W]  gaussian map of 0/1's.

'''
class PredictionHead1D(nn.Module):

    def __init__(self,\
                 segmentation_map_threshold= 0.7,\
                 gaussian_segmentation_threshold=0.7,\
                 epsilon= 1e-7):

        super(PredictionHead1D, self).__init__()
        self.segmentation_map_threshold= segmentation_map_threshold
        self.gaussian_segmentation_threshold= gaussian_segmentation_threshold
        #Pertubation to prevent gaussian from exploding
        self.epsilon= epsilon

    def create_gaussian_1D(self,\
                           variance,\
                           x_coordinate,\
                           y_coordinate,\
                           height,\
                           width, points):
       #i: indices along height
       i= torch.linspace(0,height-1,height)
       i=i.unsqueeze(1)
       i= i.repeat(1,width) # HxW
       # addede code section
       ###############################################
       # repeat on the depth
       i = i.unsqueeze(0)
       i = i.repeat(points,1,1)  # PxHxW
       ###############################################

       #j: indices along width
       j= torch.linspace(0,width-1,width)
       j=j.unsqueeze(1)
       j= j.repeat(1,height).T # HxW

       # addede code section
       ###############################################
       # repeat on the depth
       j = j.unsqueeze(0)
       j = j.repeat(points,1,1)  # PxHxW
       ###############################################

       #1D Gaussian
       A= torch.pow(i-x_coordinate,2)/(2*torch.pow(variance+self.epsilon,2))
       B= torch.pow(j-y_coordinate,2)/(2*torch.pow(variance+self.epsilon,2))
       gaussian= torch.exp(-(A+B)+ self.epsilon)

       print("gaussian shape",gaussian.shape)
       #visualization of created gaussian
       # plt.imshow(gaussian.squeeze())
       # plt.show()

       return gaussian


    def forward(self,\
                variance_map=None,\
                segmentation_map= None):


        assert variance_map is not None,\
            "variance map is none"
        assert segmentation_map is not None,\
            "segmentation map is none"
        assert variance_map.size()[1]==1,\
            "variance map should contain single channel only"
        assert segmentation_map.size()[1]==1,\
            "segmentation map should contain single channel only"

        batch_size,_,_,_=segmentation_map.size()

        #stacked output of gaussian maps
        final_stacked_gaussian_map= []

        for i in range(batch_size):

            batch_segmentation_map= segmentation_map[i,0,:,:]

            #threshold incoming segmentation_map
            batch_segmentation_map= (batch_segmentation_map>self.segmentation_map_threshold).float()

            #extract parameters for gaussians
            batch_variance= variance_map[i,0,:,:]

            #list where the gaussians of current batch are pooled
            batch_pooled_gaussians= []
            height,width= batch_variance.size()

            #gaussians will be drawn on non-zero variances return Px2 tensor.
            batch_non_zero_variances= torch.nonzero(batch_segmentation_map)
            print("non zero size",batch_non_zero_variances.shape)
            batch_no_of_non_zero_variances= batch_non_zero_variances.size()[0]


            #HANDELED SEPERATELY IN OTHER WAY IN FUTURE
            #ZERO TENSOR WILL BREAK BACKPROP GRAPH
            #handle the case when all variances are zero.
            #gaussian map will contain all zeros in this situation.
            #batch_pooled_gaussians.append(torch.zeros(height,width))
            #loop through all segmentation pixels with non zero variance
            #for drawing gaussians

            #VECTORIZE THIS CODE LATER

            # coordinate_x ==> Px1x1  ==> PxHxW
            # coordinate_y ==> Px1x1  ==> PxHxW
            # variance   ==> Px1x1    ==> PxHxW
            x_coordinate = batch_non_zero_variances[:,0]  #P
            y_coordinate = batch_non_zero_variances[:,1]  #P
            variance =   batch_variance[x_coordinate, y_coordinate] #P

            #print("type ",x_coordinate.dtype,y_coordinate.dtype,variance.dtype)
            x_coordinate = x_coordinate.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)
            y_coordinate = y_coordinate.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)
            variance = variance.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)




            batch_pooled_gaussians= self.create_gaussian_1D(variance,\
                                    x_coordinate.float(),\
                                    y_coordinate.float(),\
                                    height,\
                                    width, batch_no_of_non_zero_variances)

            #All gaussians are drawn now.
            #Take max along channel
            #batch_pooled_gaussians=torch.stack(batch_pooled_gaussians,0)
            batch_pooled_gaussians=torch.max(batch_pooled_gaussians,0).values
            #print("batch pooled gaussians",batch_pooled_gaussians.shape)
            plt.imshow(batch_pooled_gaussians)
            plt.show()
            #Gaussian thresholding
            batch_pooled_gaussians=(batch_pooled_gaussians>=self.gaussian_segmentation_threshold).float()*batch_pooled_gaussians

            #Unsqueeze to add channel dimension,
            #[1,H,W]
            batch_pooled_gaussians=batch_pooled_gaussians.unsqueeze(0)
            final_stacked_gaussian_map.append(batch_pooled_gaussians)

        #Stack Individual outputs along Common Batch dimension
        final_stacked_gaussian_map= torch.stack(final_stacked_gaussian_map,0)
        #print("shape of final map",final_stacked_gaussian_map.shape)
        return final_stacked_gaussian_map


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prediction_head=  PredictionHead1D()
    #construct synthetic input

    #segmentation present, variance present
    segmentation_map= torch.zeros((2,1,128,256))
    segmentation_map[:,0,10,10]=1
    segmentation_map[:,0,50,50]=1
    segmentation_map[:,0,100,100]=1

    variance_map= torch.zeros((2,1,128,256))
    variance_map[:,0,10,10]=27
    variance_map[:,0,50,50]=27
    variance_map[:,0,100,100]=27

    output=prediction_head(variance_map,\
                    segmentation_map)

    #segmentation present, variance 0
    #segmentation 0, variance present
    # segmentation_map= torch.zeros((2,1,128,256))
    # segmentation_map[:,0,63,63]=1
    # variance_map= torch.zeros((2,1,128,256))
    # variance_map[:,0,63,63]=0
    # output=prediction_head(variance_map,\
    #                 segmentation_map)

    #segmentation 0, variance 0
    # segmentation_map= torch.zeros((2,1,128,256))
    # segmentation_map[:,0,63,63]=0
    # variance_map= torch.zeros((2,1,128,256))
    # variance_map[:,0,63,63]=0
    # output=prediction_head(variance_map,\
    #                 segmentation_map)
    print("unique",torch.unique(output))