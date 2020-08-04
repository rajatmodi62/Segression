import os 
import numpy as np 

from util.grid_utils import convert_string_to_list,convert_list_to_string


############################### DEFAULT TESTING PARAMETERS##########################################################
#NUM_WORKERS=3
#INPUT_SIZE=512
DATASET='TOTALTEXT'
BACKBONE='VGG'
# TEST_DIR=""
TEST_DIR="data/debug/total-text-original/Images/Test"
# SEGRESSION_CHECKPOINT_PATH='snapshots/batch_size_2lr_0.0001n_steps_200000dataset_TOTALTEXTbackbone_VGG/TOTALTEXT_3d_rotated_gaussian_without_attention_10000.pth'
SEGRESSION_CHECKPOINT_PATH='snapshots/TOTALTEXT_3d_rotated_gaussian_without_attention_100000.pth'
LSTM_CHECKPOINT_PATH='snapshots/LSTM_batch_size_8lr_0.0001n_steps_100000dataset_TOTALTEXTbackbone_VGG/TOTALTEXT_LSTM_checkpoint6000.pth'
CUDA_DEVICE=0
# SEGMENTATION_THRESHOLD=0.8
# GAUSSIAN_THRESHOLD=0.65
# LSTM_THRESHOLD=0.70
N_CLASSES=3
scaling_factors= '0.25,0.50,0.75,1'
#multiply height factor by width to get the height 
height_factors=[0.3,0.6,0.9,1]
# MODE='with_lstm'
################################### GPU COMMANDS #####################################################################

#free the gpus 
os.system("nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9")

################################### UTILITY FUNCTIONS ################################################################
#returns the testing command string for subprocess in python
def create_testing_command(params):
    
    cmd= "CUDA_VISIBLE_DEVICES="+ str(CUDA_DEVICE) + " " +\
            "python -m testing_obsolete.testing_modular_lstm_rectangular" + " " +\
            "--snapshot-dir="+ str(SEGRESSION_CHECKPOINT_PATH)+ " " +\
            "--lstm-snapshot-dir="+ str(LSTM_CHECKPOINT_PATH)+ " " +\
            "--scaling_factors="+ str(scaling_factors)+ " " +\
            "--segmentation_threshold="+ str(params['segmentation_threshold'])+ " " +\
            "--gaussian_threshold="+ str(params["gaussian_threshold"])+ " " +\
            "--h_max=" + str(params["height"])+ " " +\
            "--w_max=" + str(params["width"])+ " " +\
            "--n-classes="+ str(N_CLASSES)+ " " +\
            "--test-dir="+ str(TEST_DIR)+ " " +\
            "--dataset="+ str(DATASET)+ " " +\
            "--backbone="+ str(BACKBONE)+ " " 
           
    return cmd

############################# GRID CONSTRUCTION #######################################################################

from sklearn.model_selection import ParameterGrid

param_grid = {
            'segmentation_threshold': [i/100.0 for i in range(50,100,5)],\
            'gaussian_threshold':[i/100.0 for i in range(60,100,5)],\
            'width':[i for i in range(512,512+1024,128)]
                }

grid= list(ParameterGrid(param_grid))
progress_counter=0

for grid_enumerator,params in enumerate(grid):
    #extract the parameters here 
    segmentation_threshold=params['segmentation_threshold']
    gaussian_threshold= params['gaussian_threshold']
    width= params['width']
    #for height loop 
    for height_enumerator,height_factor in enumerate(height_factors):
        params['height']= int(height_factor*width)
        cmd= create_testing_command(params)
        # print(cmd)
        os.system(cmd)

        #update the progress 
        progress_counter+=1
        progress_string=str(progress_counter)+"/"+str(len(grid)*len(height_factors))
        progress_file= './progress.txt'
        with open(progress_file, 'w') as f:
            f.write(progress_string)
        # print("segmentation_threshold",segmentation_threshold,\
        #     "gaussian_threshold",gaussian_threshold,\
        #     "width",width,\
        #     "height",height
        #     )
        
        #debugging: perform one prediction for now 
        # exit()
print("script complete!!!")
