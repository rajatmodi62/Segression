import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import random
import timeit
#from skeletonize import skeletonize_image

from packaging import version
from dataset.ctw1500 import CTW1500
from dataset.icdar2015 import ICDAR2015
from dataset.msra_td500 import MSRATD500
from dataset.total_text import TotalText
from dataset.synth_text import SynthText
from model.segression import Segression
from util.augmentation import BaseTransform, Augmentation
from loss import *
#from model.edge_detection import EdgeDetection
from loss import BinaryFocalLoss
from collections import defaultdict



start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
#     """
    parser = argparse.ArgumentParser(description="Easy TextDetection Network")
    parser.add_argument("--data-root", type=str,
                        default="/media/bashturtle/Data/Research/parimal/icip-work/important/Curved-Text-Detection/",
                        help="Absolute Dataset path")
    parser.add_argument("--batch-size", type=int, default=2,
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
    parser.add_argument("--save-pred-every", type=int, default=5000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default="./snapshots/",
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--visualization",action='store_true',
                        help="switch on the visdom based visualization .")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="checkpoint to start the training from")
    parser.add_argument("--update-visdom-iter", type=int, default= 100,
                        help="update visualization after iteration")
    parser.add_argument("--dataset", type=str, default="TOTALTEXT",
                        help="Where to save snapshots of the model.")
    parser.add_argument("--iteration-to-start-from", type=int, default= 0,
                        help="Iteration to start from if training interrupted")
    parser.add_argument("--backbone", type=str, default="VGG",
                        help="Enter the Backbone of the model BACKBONE/RESNEST")
    parser.add_argument("--train-category", type=str, default="attention",
                        help="train with attention mechanism : 'attention', 'without_attention' ")
    parser.add_argument("--out-channels", type=int, default=32,
                        help="Save summaries and checkpoint every often.")

    return parser.parse_args()

def create_snapshot_path(args):
    snapshot_dir = os.path.join(args.snapshot_dir,"batch_size_"+str(args.batch_size) + "lr_"+ str(args.learning_rate) + \
		"n_steps_"+str(args.num_steps) + "dataset_"+str(args.dataset) +"backbone_"+str(args.backbone))
    return snapshot_dir


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

    #print("param length",len(optimizer.param_groups))
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def visualization(visual_list ):

    for index in range(len(visual_list)):
        viz.image(
            visual_list[index][0],
            win=visual_list[index][1],
            opts=dict(title='TotalText Dataset', caption=visual_list[index][2]),
        )

    #
    # viz.image(
    #     visual_list[1]*100,
    #     win="ground_truth",
    #     opts=dict(title='TotalText Dataset', caption='Ground Truth for Gaussian Branch'),
    # )
    #
    # viz.image(
    #     visual_list[2].detach().cpu().numpy()*100,
    #     win="contour_map",
    #     opts=dict(title='TotalText Dataset', caption='Contour Prediction from Gaussian Branch'),
    # )
    #
    # viz.image(
    #     visual_list[3],
    #     win="score_map",
    #     opts=dict(title='TotalText Dataset', caption='Prediction for Segmentation Branch'),
    # )
    # viz.image(
    #     visual_list[4],
    #     opts=dict(title='TotalText Dataset', caption='ground truth cetner line '),
    #     win="cneter line ",
    # )
    #

    # viz.image(
    # (visual_list[3][0]),
    # win="pred_non_text",
    # opts=dict(title='TotalText Dataset', caption='Predcition Non Text'),
    # )
    # viz.image(
    # (visual_list[3][1]),
    # win="pred_border",
    # opts=dict(title='TotalText Dataset', caption='Predcition boundary pixels'),
    # )
    # viz.image(
    # (visual_list[3][2]),
    # win="pred_text",
    # opts=dict(title='TotalText Dataset', caption='Predcition Text Regions'),
    # )
    #
    # viz.image(
    #     (visual_list[4][0]*100),
    #     win="gt_non_text",
    #     opts=dict(title='TotalText Dataset', caption='GT Non Text'),
    # )
    # viz.image(
    #     (visual_list[4][1]*100),
    #     win="gt_border",
    #     opts=dict(title='TotalText Dataset', caption='GT boundary pixels'),
    # )
    # viz.image(
    #     (visual_list[4][2]*100),
    #     win="gt_text",
    #     opts=dict(title='TotalText Dataset', caption='GT Text Regions'),
    # )




def get_train_loader_object(dataset):

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
        trainset = TotalText(
            data_root=args.data_root+'/data/total-text-original',
            input_size=args.input_size,
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=args.input_size, mean=means, std=stds)
        )
    return  trainset

args = get_arguments()
print('===================>VISUALIZATION',args.visualization)
print('===================>BACKBONE',args.backbone)
if args.visualization:
    from visdom import Visdom
    viz= Visdom()

def load_model(args,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("channels are",args.out_channels)
    model=Segression(center_line_segmentation_threshold=0.999,\
                    backbone=args.backbone,\
                    segression_dimension= 3,\
                    n_classes=1,\
                    attention=False,\
                    out_channels=args.out_channels).to(device)
    if args.checkpoint!="":
        model.load_state_dict(torch.load(args.checkpoint,map_location=device),strict=False)
        print("loaded checkpoint at ",args.checkpoint)
    return model

def check_boundary_condition_and_modify(center_line_map,):
    center_w, center_h = center_line_map.shape[-2]//2, center_line_map.shape[-1]//2
    vector_of_text_presence = torch.sum(torch.sum(center_line_map.squeeze(),axis=-1),axis=-1)
    #print('vector', vector_of_text_presence)
    vector_of_non_text_presence = torch.eq(vector_of_text_presence,0)*1
    vector_of_text_presence = torch.gt(vector_of_text_presence,0)*1
    zero_indices = torch.nonzero(vector_of_non_text_presence)
    non_zero_indices = torch.nonzero(vector_of_text_presence)
    flag=True

    # check 1 : all slices of segmentation is zeros .. i.e. the training batch contains no text instances
    if torch.sum(center_line_map)==0:
        # set center pixel with value 1 i.e. put some arbitrary point on center line map
        #print("========>CASE:1")
        center_line_map[:,:,center_h, center_w]=1
        zero_indices=torch.linspace(0,center_line_map.shape[0]-1,center_line_map.shape[0]).type(torch.LongTensor).to(device)
        flag=False
        indices = zero_indices

    # check 2 : if some of the images does not contains the text instances
    elif zero_indices.shape[0]>0:
        #print("========>CASE:2")
        # set center pixel with value 1 i.e. put some arbitrary point on center line map
        #print(zero_indices)
        center_line_map[zero_indices,:,center_h, center_w]=1
        indices = non_zero_indices
    else:
        #print("========>CASE:3")
        indices=torch.linspace(0,center_line_map.shape[0]-1,center_line_map.shape[0]).type(torch.LongTensor).to(device)

    return center_line_map, indices, flag

binary_focal_loss=BinaryFocalLoss(alpha=[1.0, 0.25], gamma=2)

def find_max_batch_size_tensor(attempt = 5000,max_allowed=5000 ):
    print('find max size tensor')
    set = get_train_loader_object(args.dataset)
    loader = data.DataLoader(set, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    loader_iter = enumerate(loader)
    _, batch = loader_iter.__next__()
    max_value=0
    for i in range(attempt):
        try:
            _, batch = loader_iter.__next__()
        except:
            loader_iter = enumerate(loader)
            _, batch = loader_iter.__next__()
        image, ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map=batch
        center_line_map= center_line_map[1]
        if torch.sum(center_line_map)<=max_allowed:
        #      \
        # and torch.sum(center_line_map)>=max_allowed-peterbation:
            if max_value < torch.sum(center_line_map):
                max_value = torch.sum(center_line_map)
                best_batch=batch
        print('i   ', i,'tawa garma karne do   ',max_value)
    print('MAX NUMBER OF POINTS', max_value)
    return best_batch, max_value

def main():
    cudnn.enabled = True
    gpu = args.gpu
    #device='cpu'
    model= load_model(args,device)
    model_save_dir = create_snapshot_path(args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(model_save_dir):

        os.makedirs(model_save_dir)

    model.train()
    #edge_ = EdgeDetection().to(device)

    #Define the training superset

    trainset = get_train_loader_object(args.dataset)

    #dataset loaders
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)

    trainloader_iter = enumerate(trainloader)
    _, batch = trainloader_iter.__next__()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.99))

    optimizer.zero_grad()
    print("starting")

    iteration_to_start_from=args.iteration_to_start_from
    print("iteration to start from",iteration_to_start_from)
    max_value=torch.sum(torch.zeros(5)).to(device)

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
        adjust_learning_rate(optimizer, i_iter)
        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()

        image, ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map=batch
        if i_iter == 0 or i_iter==iteration_to_start_from:
            batch, max_value= find_max_batch_size_tensor(attempt = 500,max_allowed=5000 )
            image, ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map=batch

        if torch.sum(center_line_map[1])>max_value:
            continue
        img=image.to(device)
        # original scale as an input image
        center_line_map_orig = center_line_map[0].to(device)

        # shrined to the one fourth of the original image scale
        center_line_map=center_line_map[1].to(device)#center_line_map.to(device)
        center_line_map=center_line_map.unsqueeze(1)
        train_mask= train_mask.to(device)

        compressed_ground_truth= ground_truth[0].to(device)
        train_mask= train_mask.unsqueeze(1)
        compressed_ground_truth= compressed_ground_truth.unsqueeze(1)

        border_weight = ground_truth[2][:,1,...]*2.0

#        contour_edge_ground_truth= ground_truth[1].to(device)
        three_class_seg_ground_truth= ground_truth[3].to(device)
        # print('hello', three_class_seg_ground_truth.shape)
        # plt.subplot(1,3,1)
        # plt.imshow(three_class_seg_ground_truth[0,0,:,:].detach().cpu().numpy())
        # plt.subplot(1,3,2)
        # plt.imshow(three_class_seg_ground_truth[0,1,:,:].detach().cpu().numpy())
        # plt.subplot(1,3,3)
        # plt.imshow(three_class_seg_ground_truth[0,2,:,:].detach().cpu().numpy())
        # plt.show()

        #model.switch_gaussian_label_map(center_line_map)
        center_line_map, indices, flag = check_boundary_condition_and_modify(center_line_map)
        if max_value<torch.sum(center_line_map):
            max_value = torch.sum(center_line_map)
        #max_value=torch.max(max_value, torch.sum(center_line_map))
        #print('number of pixels', max_value,torch.sum(center_line_map))
        #print('unique elements in input center_line',torch.unique(center_line_map))

        contour_map,score_map, variance_map,meta= model(img, segmentation_map=center_line_map)

        #print('HHHHHOOOOOOOOOOOOOOOOOOOOOOOOOOO', contour_map.shape)
            #print("in train soooooovel",sobel_edge.dtype)
        #print(contour_map)
        '''
        if flag==True:
            del score_map,contour_map, flag,variance_map, image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map
            if i_iter >= args.num_steps-1:
                print ('save model ...')

                torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(args.num_steps)+'.pth'))

            if i_iter % args.save_pred_every == 0 and i_iter!=0:
                print ('taking snapshot ...')
                torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(i_iter)+'.pth'))
            continue
        '''
        contour_map=contour_map.to(device)


        contour_map=contour_map.squeeze(1)

        # center_line_map=center_line_map.squeeze(1)
        compressed_ground_truth=compressed_ground_truth.squeeze(1)

        train_mask= train_mask.squeeze(1)
        score_map=score_map.squeeze(1)
        center_line_map= center_line_map.squeeze(1)

        contour_loss_tolerance= 1
        score_map_loss_tolerance= 1
        #
        # print("dimesions",compressed_ground_truth.shape,\
        #     train_mask.shape,\
        #     score_map.shape,\
        #     center_line_map)

        # center line dice loss previous used
        loss_score_map=centre_line_dice_loss(train_mask,score_map,center_line_map_orig)


        #loss_score_map=multiclass_dice_loss(three_class_seg_ground_truth,train_mask,score_map)


        #convert train mask to bool
        train_mask= train_mask.type(torch.bool)
        # loss_score_map= binary_focal_loss(score_map,center_line_map,train_mask)
        loss_seg= score_map_loss_tolerance*loss_score_map

        #print("tst loss",loss_score_map)
        #print('loss 1', loss_seg)
        # extract non zero planes contour maps and corresponding ground truth
        loss_contour_map= -1000
        if flag:#zero_indices.shape[0]!=center_line_map.shape[0]:
            #print('perform this section ...yyyyy')
            #print(contour_map.shape, train_mask.shape, compressed_ground_truth.shape)
            #print('inside loop',indices)

            contour_map = contour_map[indices,...]
            train_mask = train_mask[indices,...]
            variance_map = variance_map[indices,...]
            compressed_ground_truth=compressed_ground_truth[indices,...]
            #hree_class_seg_ground_truth=three_class_seg_ground_truth[indices,...]
            #score_map = score_map[indices,...]
            loss_contour_map = loss_dice(train_mask,contour_map,compressed_ground_truth)

            #loss_contour_map=binary_focal_loss(contour_map,compressed_ground_truth,train_mask)
            #print("====================loss contour map",loss_contour_map)
            if args.train_category=='attention':
                variance_loss_tolerance= 0
                loss_variance = loss_mse(variance_map,compressed_ground_truth, train_mask)*variance_loss_tolerance

                #loss_edge_map=centre_line_dice_loss(contour_edge_ground_truth,train_mask,sobel_edge,contour_edge_ground_truth)
                # print('________________________________>>>>>>>>>>>>>>>>>>')
                # print('train mask ====>', train_mask)
                # print('contour edge gt ===>>', contour_edge_ground_truth)
                # print('--------->')
                #loss_edge_map = loss_dice(train_mask,sobel_edge,contour_edge_ground_truth)
                #loss_edge_map= binary_focal_loss(sobel_edge,contour_edge_ground_truth,train_mask)
                # print('uuuuuuuuuuuuuuuuuuuuuuuu ========>', loss_contour_map, loss_seg,loss_edge_map)
                #, loss_edge_map,
                #print('loss 2', loss_contour_map)
                loss_seg = loss_seg+contour_loss_tolerance*(loss_contour_map)+loss_variance
            else:
                loss_seg = loss_seg+contour_loss_tolerance*(loss_contour_map)

            if not torch.isnan(loss_seg):
                # print('hello', loss_seg)
                # print("-------------------------------------->LOSS",loss_contour_map.dtype,loss_score_map.dtype)

                loss_seg.backward()
                optimizer.step()
                i_iter+=1
            else:
                print("backprop error")
                outF = open("train_ctw_backprop_errors.txt", "a")
                outF.write(str(i_iter))
                outF.write("\n")
                outF.close()
        else:
            print("fat gaya")
            continue
        # loss_seg_value += loss_seg.data.cpu().numpy()/args.iter_size

        if i_iter%args.update_visdom_iter==0 and args.visualization:
        #if loss_contour_map==-1000 and args.visualization:
        # if 1:

            #print("visualzing the error case of segmentation",torch.unique(center_line_map))
            #print(img[0].shape,compressed_ground_truth[0,...].shape,contour_map[0].shape,score_map[0].shape,center_line_map[0].shape, variance_map[0].shape)

            visual_list = [
                           [img[0], "real_image", 'Real Image'],\
                           [compressed_ground_truth[0,...]*1.0,"ground_truth", 'Ground Truth for Gaussian Branch'],\
                           [contour_map[0]*255.0,"contour_map", 'Contour Prediction from Gaussian Branch'],\
                           [score_map[0]*1.0,"score_map", 'Prediction for Segmentation Branch'],\
                           [center_line_map_orig[0,...]*1.0,"cneter line", 'ground truth cetner line ']
                           ]

                            #three_class_seg_ground_truth[0,...]]#meta['variance_attention'][0,...]]
            #print(meta['variance_attention'].shape)
            visualization(visual_list )

        if i_iter%10==0 :
            print('exp = {}'.format(args.snapshot_dir))

            import gc
            len=0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        len+=1
                        #print(type(obj), obj.size())
                except:
                    pass
            print("len of tensors",len)
            print("backbrop counter",i_iter)
            print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_gaussian_branch = {3:.3f}, loss_segmentation_branch = {4:.3f}, loss_edge_map= {5:.3f}'.format(i_iter, args.num_steps, loss_seg,loss_contour_map,loss_score_map,loss_variance))


        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(args.num_steps)+'.pth'))

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(i_iter)+'.pth'))

        del score_map,contour_map,variance_map, image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map
        del loss_seg,loss_score_map
        if  flag:
            del loss_contour_map, loss_variance
    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
