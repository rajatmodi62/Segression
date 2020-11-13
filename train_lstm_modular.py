import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import timeit
from torch.utils import data, model_zoo
from torch.autograd import Variable
from util.augmentation import BaseTransform, Augmentation

#import feature extractor 
from model.segression import Segression

#import lstm datalodaer 
from dataset.merge_break_polygons import MergeBreakPolygons

#import lstm 
from model.lstm import BreakLine
from model.lstm import BreakLine_v2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from numba import jit, prange

# @autojit
def construct_input_parallel( args,batch_extractor, feature_map, max_length=500):
    batch_size = batch_extractor['coordinate_sequence'].shape[0]
    coordinate_sequence = batch_extractor['coordinate_sequence']
    length_vector = batch_extractor['length']
    ground_truth = batch_extractor['gt']
    valid = batch_extractor['valid']

    # print('check the argumnents ', batch_size, coordinate_sequence.shape, length_vector, ground_truth.shape)
    # input('halt')
    #  (seq_len, batch, input_size)
    input_lstm = torch.zeros(max_length, batch_size,  args.out_channels)

    for batch in prange(batch_size):
        instance_length = length_vector[batch]
        for len in prange(instance_length):
            coordinate = coordinate_sequence[batch,len,:]   # x,y
            # print("coordinates ===>",coordinate.shape)
            temp=feature_map[batch,:,coordinate[0], coordinate[1]]
            # print(temp.shape)
            input_lstm [len,batch,:] = feature_map[batch,:,coordinate[0], coordinate[1]].squeeze()
    input_lstm = Variable(input_lstm)
    ground_truth = Variable(ground_truth)    
    return  input_lstm, ground_truth, length_vector, valid 

def construct_input( args,batch_extractor, feature_map, max_length=500):
    batch_size = batch_extractor['coordinate_sequence'].shape[0]
    coordinate_sequence = batch_extractor['coordinate_sequence']
    length_vector = batch_extractor['length']
    ground_truth = batch_extractor['gt']
    valid = batch_extractor['valid']

    # print('check the argumnents ', batch_size, coordinate_sequence.shape, length_vector, ground_truth.shape)
    # input('halt')
    #  (seq_len, batch, input_size)
    input_lstm = torch.zeros(max_length, batch_size,  args.out_channels)

    for batch in range(batch_size):
        instance_length = length_vector[batch]
        for len in range(instance_length):
            coordinate = coordinate_sequence[batch,len,:]   # x,y
            # print("coordinates ===>",coordinate.shape)
            temp=feature_map[batch,:,coordinate[0], coordinate[1]]
            # print(temp.shape)
            input_lstm [len,batch,:] = feature_map[batch,:,coordinate[0], coordinate[1]].squeeze()
    input_lstm = Variable(input_lstm)
    ground_truth = Variable(ground_truth)    
    return  input_lstm, ground_truth, length_vector, valid 


def visualization(viz, output,target, length):
    max_length= torch.max(length)
    mask = torch.zeros((length.shape[0], max_length)).to(device)
    for index in range(length.shape[0]):
        mask[index,0:length[index]]=1
    viz.heatmap(
            output[:,0:max_length],
            win='prediction',
            opts=dict(title='Predictions', caption='prediction'),
        )
    viz.heatmap(
            target[:,0:max_length],
            win='target',
            opts=dict(title='Target', caption='target'),
        )

def get_train_loader_object(args):

    dataset= args.dataset

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    if dataset=='SynthText':
        trainset = SynthText(
            data_root='../SynthText',
            is_training=True,
            transform=Augmentation(size=args.input_size, mean=means, std=stds)
        )
    elif dataset=='CTW1500':
        trainset = CTW1500(
            data_root=args.data_root+'/data/ctw-1500-original',
            input_size=args.input_size,
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=args.input_size, mean=means, std=stds)
        )
    elif dataset=='MSRATD500':
        trainset = MSRATD500(
            data_root=args.data_root+'/data/msra-td500',
            input_size=args.input_size,
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=args.input_size, mean=means, std=stds)
        )
    elif dataset=='ICDAR2015':
        trainset = ICDAR2015(
            data_root=args.data_root+'/data/icdar-2015',
            input_size=args.input_size,
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=args.input_size, mean=means, std=stds)
        )
    elif dataset=='TOTALTEXT':
        trainset = MergeBreakPolygons(
            data_root=args.data_root+'/data/total-text',
            input_size=args.input_size,
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=args.input_size, mean=means, std=stds)
        )
	
    return  trainset

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(args,optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

    #print("param length",len(optimizer.param_groups))
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
#     """
    parser = argparse.ArgumentParser(description="LSTM")
    parser.add_argument("--data-root", type=str,
                        default="./",
                        help="Absolute Dataset path")
    parser.add_argument("--backbone", type=str, default="VGG",
                        help="Enter the Backbone of the model BACKBONE/RESNEST")

    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=3,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--input-size", type=int, default=512,
                        help="Input size of the model.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-steps", type=int, default=100000,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=1000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default="./snapshots/",
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--visualization",action='store_true',
                        help="switch on the visdom based visualization .")
    
    parser.add_argument("--update-visdom-iter", type=int, default= 100,
                        help="update visualization after iteration")
    parser.add_argument("--dataset", type=str, default="TOTALTEXT",
                        help="Where to save snapshots of the model.")
    parser.add_argument("--iteration-to-start-from", type=int, default= 0,
                        help="Iteration to start from if training interrupted")
    parser.add_argument("--out-channels", type=int, default=32,
                        help="Save summaries and checkpoint every often.")
    # parser arguments relevant to this code 

    parser.add_argument("--feature_extractor_checkpoint", type=str, default="",
                        help="enter the checkpoint for feature extractor backbone")

    
    return parser.parse_args()

def create_snapshot_path(args):
    snapshot_dir = os.path.join(args.snapshot_dir,"LSTM_"+"batch_size_"+str(args.batch_size) + "lr_"+ str(args.learning_rate) + \
		"n_steps_"+str(args.num_steps) + "dataset_"+str(args.dataset) +"backbone_"+str(args.backbone))
    return snapshot_dir

def load_feature_extractor_model(args):
    # model=Segression(center_line_segmentation_threshold=0.999,\
    #                 backbone=args.backbone,\
    #                 segression_dimension= 3,\
    #                 n_classes=3,\
    #                 attention=False,\
    #                 out_channels=args.out_channels).to(device)
    model= Segression(center_line_segmentation_threshold=0.999,\
                    backbone=args.backbone,\
                    segression_dimension= 3,\
                    out_channels= args.out_channels,\
                    n_classes=3,\
                    mode='test').to(device)

    print("loading state dictionary of feature extractor at:",args.feature_extractor_checkpoint)
    model.load_state_dict(torch.load(args.feature_extractor_checkpoint,map_location=device),strict=True)
    print("backbone loading successful!!!")
    return model

def compute_binary_cross_entropy(output,target, length, valid):

    #input batch x seq_length
    criterion = nn.BCELoss(reduction='mean')
    total_loss=0
    for index in range(output.shape[0]):
        if valid[index]==1:
            total_loss+= criterion(output[index,0:length[index]], target[index,0:length[index]]) 
    return total_loss

def compute_binary_cross_entropy_v2(output,target, mask):
    #print('shapes', output.shape, target.shape, mask.shape)
    #input batch x seq_length
    criterion = nn.BCELoss(reduction='none')
    total_loss= criterion(output.squeeze(), target.squeeze()) 
    mask = torch.gt(mask.squeeze(),0)
    total_loss = torch.masked_select(total_loss,mask)
    total_loss = torch.mean(total_loss)
    return total_loss

def dice_loss(probas,target):
    eps=1e-7
    intersection= torch.sum(probas*target)
    cardinality = torch.sum(probas + target)
    dice_loss = (2. * intersection / (cardinality + eps))
    loss = (1 - dice_loss)
    return loss

def compute_dice_loss(output,target, length, valid):
    #input batch x seq_length
    
    total_loss=0
    for index in range(output.shape[0]):
        if valid[index]==1:
            total_loss+= dice_loss(output[index,0:length[index]], target[index,0:length[index]]) 
    return total_loss

def load_lstm_model(args):
    
    model= BreakLine(number_of_layer=1, batch=args.batch_size, hidden_size=64, num_directions=1).to(device)
    # print("loading lstm checkpoint")    
    # model.load_state_dict(torch.load("snapshots/LSTM_batch_size_8lr_0.0001n_steps_100000dataset_TOTALTEXTbackbone_VGG/TOTALTEXT_LSTM_checkpoint3000.pth",\
    #                         map_location=device),strict=True)
    print("============================loaded lstm ")
    return model

def trainer_v1():
    args = get_arguments()
    if args.visualization:
        from visdom import Visdom
        viz= Visdom()
    
    cudnn.enabled = True
    gpu = args.gpu
    model_feature_extractor= load_feature_extractor_model(args)
    model_lstm = load_lstm_model(args)

    model_save_dir = create_snapshot_path(args)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(model_save_dir):

        os.makedirs(model_save_dir)
    
    model_feature_extractor.eval()
    model_lstm.train()

    #Define the training superset

    trainset = get_train_loader_object(args)

    #dataset loaders
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,drop_last=True)
    criterion = nn.BCELoss(reduction='mean')
    trainloader_iter = enumerate(trainloader)
    _, batch = trainloader_iter.__next__()
    optimizer = optim.Adam(model_lstm.parameters(), lr=args.learning_rate, betas=(0.9,0.99))

    optimizer.zero_grad()
    print("starting")

    iteration_to_start_from=args.iteration_to_start_from
    print("iteration to start from",iteration_to_start_from)

    #for i_iter in range(iteration_to_start_from,args.num_steps):
    i_iter=0
    while 1:
        if i_iter>args.num_steps:
            break
        if i_iter<iteration_to_start_from:
            if i_iter%1000==0:
                print("skipping till ",i_iter,iteration_to_start_from)
            i_iter+=1
            continue
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(args,optimizer, i_iter)
        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()

        with torch.no_grad():
            #print("extracting feature",batch['image'].shape,type(batch['image']))
            #batch['image'] = batch['image'].transpose(2, 0, 1)
            batch['image']=batch['image'].to(device)
            _,_,feature = model_feature_extractor(batch['image'],segmentation_map=None)
            #print("Babua!!!! Feature extract ho gaya ",feature.shape)
        input_lstm, target, length_vector, valid  = construct_input_parallel( args,batch, feature, max_length=500)
        print('batch size,', length_vector.shape)
        if torch.sum(valid)>0:
            input_lstm=input_lstm.to(device)
            target = target.to(device)
            target= target.float()
            #print("input",input_lstm.shape)
            output = model_lstm(input_lstm)
            #print('shapes', output.shape, target.shape)
            loss = compute_dice_loss(output, target, length_vector, valid  ) 
            loss.backward()
            optimizer.step()
            i_iter+=1
            if i_iter%10==0:
                print("i:iter",i_iter,"loss:",loss)
                if args.visualization:
                    visualization(viz,output,target,length_vector)

def main():
    args = get_arguments()
    if args.visualization:
        from visdom import Visdom
        viz= Visdom()
    
    cudnn.enabled = True
    gpu = args.gpu
    model_feature_extractor= load_feature_extractor_model(args)
    #model_lstm = load_lstm_model(args)

    model_save_dir = create_snapshot_path(args)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(model_save_dir):

        os.makedirs(model_save_dir)
    
    model_feature_extractor.eval()
    hidden_size=64

    lstm_model = BreakLine_v2( hidden_size=hidden_size,output_size=1,feature_feature=32).cuda()
    # print("loading lstm checkpoint")    
    # lstm_model.load_state_dict(torch.load("snapshots/LSTM_batch_size_8lr_0.0001n_steps_100000dataset_TOTALTEXTbackbone_VGG/TOTALTEXT_LSTM_checkpoint1000.pth",\
    #                         map_location=device),strict=True)
    print("============================loaded lstm ")
    lstm_model.train()

    #Define the training superset

    trainset = get_train_loader_object(args)

    #dataset loaders
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,drop_last=True)
    criterion = nn.BCELoss(reduction='mean')
    trainloader_iter = enumerate(trainloader)
    _, batch = trainloader_iter.__next__()
    optimizer = optim.Adam(lstm_model.parameters(), lr=args.learning_rate, betas=(0.9,0.99))
   
    optimizer.zero_grad()
    print("starting")

    iteration_to_start_from=args.iteration_to_start_from
    print("iteration to start from",iteration_to_start_from)

    #for i_iter in range(iteration_to_start_from,args.num_steps):
    i_iter=0
    while 1:
        hidden = lstm_model.initHidden(args.batch_size)
        if i_iter>args.num_steps:
            break
        if i_iter<iteration_to_start_from:
            if i_iter%1000==0:
                print("skipping till ",i_iter,iteration_to_start_from)
            i_iter+=1
            continue
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(args,optimizer, i_iter)
        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()

        with torch.no_grad():
            #print("extracting feature",batch['image'].shape,type(batch['image']))
            #batch['image'] = batch['image'].transpose(2, 0, 1)
            batch['image']=batch['image'].to(device)
            _,_,feature = model_feature_extractor(batch['image'],segmentation_map=None)
            #print("Babua!!!! Feature extract ho gaya ",feature.shape)
        input_lstm, target, length_vector, valid  = construct_input( args,batch, feature, max_length=500)
        batch_size=length_vector.shape
        max_length= torch.max(length_vector)
        
        mask = torch.zeros((length_vector.shape[0], max_length)).to(device)
        for index in range(length_vector.shape[0]):
            mask[index,0:length_vector[index]]=1


        if torch.sum(valid)>0:
            input_lstm=input_lstm.to(device)
            target = target.to(device)
            target= target.float()
            #print("input",input_lstm.shape)
            
            output = torch.ones(8, 1, device=device)
            loss=0
            stacked_output = []
            for index in range(max_length):
                if index==0:
                    output, hidden = lstm_model(input_lstm[index,...].cuda(), hidden.cuda(), output.cuda())
                    #print('output shape', output.shape)
                else:
                    #sample with a uniform distribution of 0.5 
                    probability= np.random.uniform(0,1)
                    if probability>0.5:
                       # print("with teacher forcing")
                        teacher=target[:,index-1].unsqueeze(1)
                        #print('teacher', teacher.shape)
                        output, hidden = lstm_model(input_lstm[index,...].cuda(), hidden.cuda(), teacher.cuda())
                    else:
                        #print("without teacher forcing")
                        output=(output>0.8)*1.0
                        output, hidden = lstm_model(input_lstm[index,...].cuda(), hidden.cuda(), output.cuda())
                loss+=compute_binary_cross_entropy_v2(output, target[:,index], mask[:,index] ) 
                stacked_output.append(output.squeeze().detach().unsqueeze(1))

            stacked_output=torch.cat(stacked_output,dim=1)
            #print(stacked_output.shape, target.shape)
            loss.backward()
            optimizer.step()
            i_iter+=1
            if i_iter%10==0:
                print("i:iter",i_iter,"loss:",loss)
            if args.visualization and i_iter%100==0:
                visualization(viz,stacked_output,target,length_vector)
        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(lstm_model.state_dict(),osp.join(model_save_dir, args.dataset+'_LSTM_checkpoint'+str(args.num_steps)+'.pth'))

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(lstm_model.state_dict(),osp.join(model_save_dir, args.dataset+'_LSTM_checkpoint'+str(i_iter)+'.pth'))


main()