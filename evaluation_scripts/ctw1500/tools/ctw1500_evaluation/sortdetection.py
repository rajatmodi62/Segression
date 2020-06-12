import os
import numpy as np
import argparse


#delete sorted and cache files
if os.path.exists('evaluation_scripts/ctw1500/tools/ctw1500_evaluation/annots.pkl'):
    os.remove('evaluation_scripts/ctw1500/tools/ctw1500_evaluation/annots.pkl')

if os.path.exists('evaluation_scripts/ctw1500/tools/ctw1500_evaluation/detections_text0.5.txt'):
    os.remove('evaluation_scripts/ctw1500/tools/ctw1500_evaluation/detections_text0.5.txt')

parser = argparse.ArgumentParser(description="Welcome to CTW evaluation!!")

parser.add_argument("--input-dir", type=str, default='results/CTW1500gaussian_threshold=0.6_segmentation_threshold=0.4',
                    help="Directory where TOTALTEXT predictions are kept")
args = parser.parse_args()

#enter the prediction paths in annotation_path
anno_path = args.input_dir
outputstr = "evaluation_scripts/ctw1500/tools/ctw1500_evaluation/detections_text"
# score_thresh_list=[0.2, 0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8, 0.9]
score_thresh_list=[0.5]
files = os.listdir(anno_path)
files.sort()
for iscore in score_thresh_list:
    with open(outputstr+str(iscore)+'.txt', "w") as f1:
        for ix, filename in enumerate(files):
            print(filename)
            imagename = filename[:-4]
            print(imagename)

            with open(os.path.join(anno_path, filename), "r") as f:
                lines = f.readlines()

            for line in lines:
                box = line.strip().split(",")
                assert(len(box) %2 == 0) ,'mismatch xy'
                out_str = "{} {}".format(str(int(imagename[:])-1001), 0.999)
                for i in box:
                    out_str = out_str+' '+str(i)
                f1.writelines(out_str + '\n')
