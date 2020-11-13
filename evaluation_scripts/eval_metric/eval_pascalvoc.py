import os
import numpy as np
from iou import DetectionIoUEvaluator
from collections import defaultdict
import argparse
from datetime import date

def get_arguments():
	"""Parse all the arguments provided from the CLI.
	Returns:
	A list of parsed arguments.
	#     """
	parser = argparse.ArgumentParser(description="Easy TextDetection Network")
	parser.add_argument("--prediction-dir", type=str,
                        default="./prediction/",
                        help="directory contains predictions")
	parser.add_argument("--groundtruth-dir", type=str,
                        default="./gt/totaltext/test_gts",
                        help="directory contains ground truth")
	parser.add_argument("--scaling-list", type=str,
                        default="[512,512+128,512+2*128+ 512*3*128]",
                        help="list of scales")
	return parser.parse_args()

def files_with_extension(path, ext):
	file_list=[]
	for file in os.listdir(path):
		if file.endswith(ext):
			file_list.append(os.path.join(path, file))
	return file_list

def create_dictionary_for_gt(filename, type='quad'):

	gt= defaultdict(dict)
	with open(filename) as f:
		content = f.readlines()
	for index in range(len(content)):
		line = content[index].strip().split(',')
		if type=='quad':
			coord = np.asarray(line[0:8],dtype=int).reshape(-1,2)
			text = line[8]
			tag=False
			if text =='###':
				tag=True
		else:
			coord = np.asarray(line[:-1],dtype=int).reshape(-1,2)
			text = line[-1]
			tag=False
			if text =='###':
				tag=True

		gt[index]['points']=coord
		gt[index]['text']=text
		gt[index]['ignore']= tag
	return gt

def create_dictionary_for_pred(filename, type='with_confidence'):

	pred= defaultdict(dict)
	with open(filename) as f:
		content = f.readlines()
	for index in range(len(content)):
		line = content[index].strip().split(',')
		if type=='with_confidence':
			coord = np.asarray(line[:-1],dtype=int).reshape(-1,2)
		else:
			coord = np.asarray(line,dtype=int).reshape(-1,2)

		text = '1234'
		tag=False
		pred[index]['points']=coord
		pred[index]['text']=text
		pred[index]['ignore']= tag
	return pred

def main():
	args = get_arguments()
	evaluator = DetectionIoUEvaluator()
	gt_path=args.groundtruth_dir
	pred_path=args.prediction_dir
	gaussian_threshold = "0.5"#pred_path.split(os.sep)[-1].split('_')[1].split("=")[-1]
	today = date.today()
	print("========================================================\n\
DATE=",today)
	print(pred_path, gt_path)
	segmentation_threshold = "0.5"#pred_path.split(os.sep)[-1].split('_')[-1].split("=")[-1]

	file_list_gt = files_with_extension(gt_path, '.txt')
	raw_metric=[]
	print("SEGMENTATION_THRESHOLD="+segmentation_threshold+" \n\
GAUSSIAN THRESHOLD ="+ gaussian_threshold+" \n\
multiscale :"+args.scaling_list+" \n\
========================================================\n")
	for i,file_gt in enumerate(file_list_gt):
		print('filename ', file_gt, i,len(file_list_gt))
		#file_pred = os.path.join(pred_path,'res_'+file_gt.split(os.sep)[-1])
		#total-text
		# file_pred = os.path.join(pred_path,'res_'+file_gt.split('/')[-1].split('_')[-1])
		#icdar-2015
		file_pred = os.path.join(pred_path,'res_'+file_gt.split('/')[-1][3:])
		print("file pred",file_pred)

		print("rajat",file_gt.split('/')[-1].split('_')[-1])
		gt = create_dictionary_for_gt(file_gt, type='poly')
		pred = create_dictionary_for_pred(file_pred,type='without_confidence')
		results = []
		metric = evaluator.evaluate_image(gt, pred)
		print('Precision:', metric['precision'],'Recall', metric['recall'],'hmean', metric['hmean'])
		raw_metric.append(metric)
	result = evaluator.combine_results(raw_metric)
	print(result)

if __name__=='__main__':
	main()
