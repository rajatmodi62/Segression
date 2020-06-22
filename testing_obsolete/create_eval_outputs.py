import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil as sh
from pathlib import Path


class EvalOutputs():
    def __init__(self,\
                args= None,\
                ):
        super(EvalOutputs, self).__init__()

        #initialize paths
        self.local_data_path = Path('.').absolute()

        #store dataset name
        self.dataset= args.dataset
        self.dataset_dir=(self.local_data_path/'results'/(self.dataset+\
                            'gaussian_threshold='+str(args.gaussian_threshold)+\
                            '_segmentation_threshold='+ str(args.segmentation_threshold)))
        # #delete existing predictions
        if os.path.isdir(str(self.dataset_dir)):
            sh.rmtree(str(self.dataset_dir))
        #create dir containing the dataset
        (self.dataset_dir).mkdir(exist_ok=True, parents=True)

    '''
    Swap=False dump x, y predictions
    Swap= True dums y,x predictions
    '''
    def generate_x_y_output(self,contour_list,image_id,swap=False):
        #strip image_id and save as txt
        image_id= image_id.split('.')[0]+'.txt'
        print('============>IMAGE ID',image_id)
        pred_path= str(self.dataset_dir/('res_'+image_id))
        fid = open(pred_path, 'a')
        content=''

        for contour in contour_list:

            #skip the contours that dont satisfy the area threshold
            if contour=='':
                continue
            rows,cols= contour.shape
            for row in range(rows):

                item=contour[row,:]
                x= item[0]
                y= item[1]
                if swap:
                    content+=str(y)+','+str(x)+','
                else:
                    content+=str(x)+','+str(y)+','
            #remove last comma

            content= content[:-1]+'\n'
            fid.write(content)
            #reset
            content=''
        fid.close()

    '''
    Input:
        contour_list: list of contours. [Numpy arrays <row,2>]
                    Each row represents x,y coordinate of a point
    NOTE: contour_list must be scaled back to the original image size before call.
    '''


    def generate_predictions(self,contour_list,image_id):
        #check dataset & accordingly process
        if self.dataset=='CTW1500':

            self.generate_x_y_output(contour_list,image_id,swap=False)
            # write command for executing the evaluation script
            #os.system('python ')

        elif self.dataset=='TOTALTEXT':

            self.generate_x_y_output(contour_list,image_id,swap=False)
            # write command for executing the evaluation script
            #os.system('python Deteval.py ')

        elif self.dataset=='MSRATD500':

            #to handle
            raise Exception("Prediction Error:Invalid Dataset given")

        elif self.dataset=='ICDAR2015':
            self.generate_x_y_output(contour_list,'res_'+image_id,swap=False)
            #to handle
            #raise Exception("Prediction Error:Invalid Dataset given")

        else:
            raise Exception("Prediction Error:Invalid Dataset given")

#EvalOutputs(dataset="Bismillah")
