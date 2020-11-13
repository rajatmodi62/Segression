import numpy as np
import os
import subprocess
import shutil as sh
from pathlib import Path
from evaluation_scripts.totaltext.Deteval import Deteval
from collections import defaultdict
import json

os.system("nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9")

segmentation_thresholds= [0.40,0.50,0.60,0.70,0.80,0.90,0.95]
gaussian_thresholds= [0.60,0.65,0.70,0.75,0.85,0.90,0.95]

dataset='TOTALTEXT'
backbone='DB'
snapshot_dir='snapshots/batch_size_4lr_0.0001n_steps_100000dataset_TOTALTEXTbackbone_DB/TOTALTEXT_3d_rotated_gaussian_without_attention_10000.pth'
out_channels= 256

local_data_path = Path('.').absolute()
local_data_path.mkdir(exist_ok=True)
#scores
scores= defaultdict(dict)
best_f1_till_now=-1
i=0
for segmentation_threshold in segmentation_thresholds:
    for gaussian_threshold in gaussian_thresholds:
        #create the testing command
        testing_cmd= 'python -m testing_obsolete.testing_square.py --backbone '+ str(backbone)+\
                        ' --snapshot-dir ' + str(snapshot_dir) +\
                        ' --out-channels ' + str(out_channels) +\
                        ' --gaussian_threshold ' + str(gaussian_threshold) +\
                        ' --segmentation_threshold ' + str(segmentation_threshold)
        os.system(testing_cmd)
        result_dir= (local_data_path/'results'/(dataset+\
                            'gaussian_threshold='+str(gaussian_threshold)+\
                            '_segmentation_threshold='+ str(segmentation_threshold)))
        p,r,f= Deteval(result_dir)

        if f>best_f1_till_now:
            best_f_till_now=f
        print("===================>best f1 is ",best_f1_till_now)
        #dump to dictionary
        scores[i]['segmentation_threshold']=segmentation_threshold
        scores[i]['gaussian_threshold']=gaussian_threshold
        scores[i]['precision']=p
        scores[i]['recall']=r
        scores[i]['f1']=f
        i+=1
        #print(result_dir)
        #exit()
        if os.path.exists('./testing_obsolete/results.json'):
            os.remove('./testing_obsolete/results.json')
        #print("type",type(scores))
        content = json.dumps(scores)
        f = open("./testing_obsolete/results.json","w")
        f.write(content)
        f.close()

print("rigourous done")
print("best f score found at",best_f_till_now)
