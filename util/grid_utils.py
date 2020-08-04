import os 
import numpy as np 

def convert_string_to_list(input_str):
  list= [eval(i) for i in input_str.split(',')]
  return list

def convert_list_to_string(list):
  input_str= ""
  for item in list:
    print("item",item)
    input_str+=str(item)
    input_str+=","
  input_str=input_str[:-1]
  return input_str  

def create_config_dict(dataset=None,\
                      test_dir=None,\
                      snapshot_dir=None,\
                      segmentation_threshold=None,\
                      gaussian_threshold=None,\
                      backbone=None,\
                      out_channels=None,\
                      n_classes=None,\
                      h_max=None,\
                      w_max=None,\
                      scaling_factors=None,\
                      hardcode=None,\
                      hardcoded_scales=None):

  dict={}
  dict['dataset']=dataset
  dict['test_dir']=test_dir
  dict['snapshot_dir']=snapshot_dir
  dict['segmentation_threshold']=segmentation_threshold
  dict['gaussian_threshold']=gaussian_threshold 
  dict['backbone']=backbone
  dict['out_channels']=out_channels
  dict['n_classes']=n_classes
  dict['h_max']=h_max
  dict['w_max']=w_max
  dict['scaling_factors']=scaling_factors
  dict['hardcode']=hardcode
  dict['hardcoded_scales']=hardcoded_scales

  return dict