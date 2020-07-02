INPUT_DIR='./results/CTW1500gaussian_threshold=0.65_segmentation_threshold=0.99/'
cd $INPUT_DIR
pwd
rename 's/res_//' *.txt
cd -
pwd
rm -f evaluation_scripts/ctw1500/tools/ctw1500_evaluation/detections_text0.5.txt 
rm -f evaluation_scripts/ctw1500/tools/ctw1500_evaluation/annots.pkl
python evaluation_scripts/ctw1500/tools/ctw1500_evaluation/sortdetection.py --input-dir $INPUT_DIR
python evaluation_scripts/ctw1500/tools/ctw1500_evaluation/test_ctw1500_eval.py
