import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from model.edge_detection import EdgeDetection

def get_slice(total_indices,size_of_slice):
  slice=[]
  if total_indices%size_of_slice==0:
    no_of_slices= int(total_indices//size_of_slice)
    for idx in range(no_of_slices):
      slice.append((idx*size_of_slice,(idx+1)*size_of_slice))
  else:
    no_of_slices= int(total_indices/size_of_slice) + 1
    for idx in range(no_of_slices-1):
      slice.append((idx*size_of_slice,(idx+1)*size_of_slice))
    slice.append(((no_of_slices-1)*size_of_slice,total_indices))

  return slice

'''
Prediction Head takes a segmentation map as input.
Variance Map for segmented pixels is used to create a gaussian.
Gaussians cascaded, max pooled & thresholded.

Input:
Variance Map: [N,3,H,W]
Segmentation Map: [N,1,H,W] tensor of 0/1's

Parameters:
Gaussian Kernel containing a single variance.

Returns:
[1,1,H,W]  gaussian map of [0,1]

'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PredictionHead3D(nn.Module):

    def __init__(self,\
                 segmentation_map_threshold= 0.7,\
                 gaussian_segmentation_threshold=0.7,\
                 epsilon= 1e-7):

        super(PredictionHead3D, self).__init__()
        self.segmentation_map_threshold= segmentation_map_threshold
        self.gaussian_segmentation_threshold= gaussian_segmentation_threshold
        #Pertubation to prevent gaussian from exploding
        self.epsilon= epsilon

        #self.epsilon=0
    def create_gaussian_3D(self,\
                           variance_height,\
                           variance_width,\
                           theta,\
                           x_coordinate,\
                           y_coordinate,\
                           height,\
                           width):
        ###############################################
        #rotated gaussians
        #define the angular values
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        variance_height = torch.pow(variance_height,2)
        variance_width = torch.pow(variance_width,2)
        # a= (cos*cos)/(2*variance_height+self.epsilon) + (sin*sin)/(2*variance_width+self.epsilon)
        # b= (-2*sin*cos)/(4*variance_height+self.epsilon) + (2*sin*cos)/(4*variance_width+self.epsilon)
        # c= (sin*sin)/(2*variance_height+self.epsilon) + (cos*cos)/(2*variance_width+self.epsilon)
        a= (cos*cos)/(2*variance_height) + (sin*sin)/(2*variance_width)
        b= (-2*sin*cos)/(4*variance_height) + (2*sin*cos)/(4*variance_width)
        c= (sin*sin)/(2*variance_height) + (cos*cos)/(2*variance_width)
        # print("==========>a",torch.min(a),torch.max(a))
        # print("=========>b",torch.min(b),torch.max(b))
        # print('variance height',torch.min(variance_height),torch.max(variance_height))
        # print('variance height',torch.min(variance_height),torch.max(variance_width))
        # print('c',torch.max(sin))
        # print('d', torch.max(cos))

        ###############################################

        #i: indices along height
        i= torch.linspace(0,height-1,height).to(device)
        i=i.unsqueeze(1)
        i= i.repeat(1,width) # HxW


        #j: indices along width
        j= torch.linspace(0,width-1,width).to(device)
        j=j.unsqueeze(1)
        j= j.repeat(1,height).T # HxW



        #the x & y around which gaussian is centered
        A= torch.pow(i-x_coordinate,2)
        B= 2*(i-x_coordinate)*(j-y_coordinate)
        C= torch.pow(j-y_coordinate,2)
        gaussian= torch.exp(-(a*A+b*B+c*C)+self.epsilon)#.cuda()

        #print('max values of gaussian', torch.max(gaussian))


        # #1D Gaussian
        # A= torch.pow(i-x_coordinate,2)/(2*torch.pow(variance_height+self.epsilon,2))
        # B= torch.pow(j-y_coordinate,2)/(2*torch.pow(variance_width+self.epsilon,2))
        # gaussian= torch.exp(-(A+B)+ self.epsilon)

        #print("gaussian shape",gaussian.shape)
        #visualization of created gaussian
        # plt.imshow(gaussian.squeeze())
        # plt.show()
        del a,b,c,A,B,C,i,j
        # return torch.max(gaussian,0).values

        return gaussian


    def forward(self,\
                variance_map=None,\
                segmentation_map= None,size_of_slice=200):


        assert variance_map is not None,\
            "variance map is none"
        assert segmentation_map is not None,\
            "segmentation map is none"
        assert variance_map.size()[1]==3,\
            "variance map should contain three channel only"
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
            # Added 1 perturbation to prevent explosion
            batch_variance_height= F.relu(variance_map[i,0,:,:])+1
            batch_variance_width= F.relu(variance_map[i,1,:,:])+ 1
            # batch_theta_map= 3.14*variance_map[i,2,:,:]

            batch_theta_map= 3.14*F.sigmoid(variance_map[i,2,:,:])

            #list where the gaussians of current batch are pooled
            batch_pooled_gaussians= []
            height,width= batch_variance_height.size()

            #gaussians will be drawn on non-zero variances return Px2 tensor.
            batch_non_zero_variances= torch.nonzero(batch_segmentation_map)
            #print("non zero size",batch_non_zero_variances.shape)
            batch_no_of_non_zero_variances= batch_non_zero_variances.size()[0]

            for j in range(batch_no_of_non_zero_variances):
                x_coordinate=batch_non_zero_variances[j,0].item()
                y_coordinate= batch_non_zero_variances[j,1].item()


                # #print("type ",x_coordinate.dtype,y_coordinate.dtype,variance.dtype)
                # x_coordinate = x_coordinate.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)
                # y_coordinate = y_coordinate.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)
                # variance_height = variance_height.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)
                # variance_width = variance_width.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)
                # theta_map = theta_map.view(batch_no_of_non_zero_variances,1,1).repeat(1,height,width)


                batch_gaussians= self.create_gaussian_3D(batch_variance_height[x_coordinate][y_coordinate],\
                                        batch_variance_width[x_coordinate][y_coordinate],\
                                        batch_theta_map[x_coordinate][y_coordinate],\
                                        x_coordinate,\
                                        y_coordinate,\
                                        height,\
                                        width)
                #print("batch gaussians shape",batch_gaussians.shape)
                batch_pooled_gaussians.append(batch_gaussians)

            #All gaussians are drawn now.
            #Take max along channel
            batch_pooled_gaussians=torch.stack(batch_pooled_gaussians,0)
            batch_pooled_gaussians=torch.max(batch_pooled_gaussians,0).values


            #Gaussian thresholding
            batch_pooled_gaussians=(batch_pooled_gaussians>=self.gaussian_segmentation_threshold).float()*batch_pooled_gaussians
            # if type=='border':
            #     batch_pooled_gaussians=self.edge(batch_pooled_gaussians)

            #Unsqueeze to add channel dimension,
            #[1,H,W]
            #print("shape of batch pooled gaussians",batch_pooled_gaussians.shape)
            batch_pooled_gaussians=batch_pooled_gaussians.unsqueeze(0)
            final_stacked_gaussian_map.append(batch_pooled_gaussians)

        #Stack Individual outputs along Common Batch dimension
        final_stacked_gaussian_map= torch.stack(final_stacked_gaussian_map,0)
        #print("shape of final map",final_stacked_gaussian_map.shape)
        return final_stacked_gaussian_map


if __name__ == '__main__':

    prediction_head=  PredictionHead3D()
    #construct synthetic input

    #segmentation present, variance present
    segmentation_map= torch.zeros((2,1,128,256))
    segmentation_map[:,0,10,10]=1
    # segmentation_map[:,0,50,50]=1
    # segmentation_map[:,0,100,100]=1

    variance_map= torch.zeros((2,3,128,256))
    variance_map[:,0,10,10]=0
    variance_map[:,1,10,10]=0
    variance_map[:,2,10,10]=0

    # variance_map[:,0,50,50]=100
    # variance_map[:,1,50,50]=27
    # variance_map[:,2,10,10]=0
    #
    # variance_map[:,0,100,100]=100
    # variance_map[:,1,100,100]=27
    # variance_map[:,2,10,10]=0

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
    #print("unique",torch.unique(output))
