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
from model.train_east_gaussian_rotated import Segression
from util.augmentation import BaseTransform, Augmentation
from loss import *

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

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
    parser.add_argument("--backbone", type=str, default="vgg",
                        help="Enter the Backbone of the model BACKBONE/RESNEST")
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
    viz.image(
        visual_list[0],
        win="real_image",
        opts=dict(title='TotalText Dataset', caption='Real Image'),
    )
    viz.image(
        visual_list[1]*100,
        win="ground_truth",
        opts=dict(title='TotalText Dataset', caption='Ground Truth for Gaussian Branch'),
    )
    viz.image(
        visual_list[2][0].detach().cpu().numpy()*100,
        win="contour_map",
        opts=dict(title='TotalText Dataset', caption='Contour Prediction from Gaussian Branch'),
    )
    viz.image(
        visual_list[3],
        win="score_map",
        opts=dict(title='TotalText Dataset', caption='Prediction for Segmentation Branch'),
    )
    viz.image(
        (visual_list[4]),
        win="center_line_map",
        opts=dict(title='TotalText Dataset', caption='Ground Truth for Segmentation Branch'),
    )
    viz.image(
        (visual_list[5].detach().cpu().numpy()),
        win="changed_variance_map",
        opts=dict(title='TotalText Dataset', caption='Gaussian Variance Map  conditioning Segmentation Branch'),
    )


def get_train_loader_object(dataset):
    #trainset = getattr('dataset', dataset)(
        #     data_root='data/ctw-1500',
        #     input_size=args.input_size,
        #     ignore_list=None,
        #     is_training=True,
        #     transform=Augmentation(size=args.input_size, mean=means, std=stds)
        # )
    #define the mean & std
    #compatible with ImageNet trained ObjectRecog models
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
            data_root=args.data_root+'/data/ctw-1500',
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
            data_root=args.data_root+'/data/total-text',
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

    model=Segression(segmentation_threshold=0.999,backbone=args.backbone).to(device)
    if args.checkpoint!="":
        model.load_state_dict(torch.load(args.checkpoint,map_location=device),strict=False)
        print("loaded checkpoint at ",args.checkpoint)
    return model


	
def main():
    cudnn.enabled = True
    gpu = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= load_model(args,device)
    model_save_dir = create_snapshot_path(args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model.train()
    


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

    for i_iter in range(iteration_to_start_from,args.num_steps):
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()

        image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map=batch
        img=image.to(device)
        center_line_map=center_line_map.to(device)
        center_line_map=center_line_map.unsqueeze(1)
        train_mask= train_mask.to(device)
        compressed_ground_truth= compressed_ground_truth.to(device)
        train_mask= train_mask.unsqueeze(1)
        compressed_ground_truth= compressed_ground_truth.unsqueeze(1)



        '''
        plt.imshow(compressed_ground_truth[0,...].numpy())
        plt.show()
        plt.imshow(center_line_map[0,...].numpy())
        plt.show()
        '''


        model.switch_gaussian_label_map(center_line_map)
        #print("uniqye",center_line_map.size(),torch.unique(center_line_map),torch.sum(center_line_map))
        score_map,contour_map, flag,variance_map=model(img)

        if flag==True:
            del score_map,contour_map, flag,variance_map, image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map
            if i_iter >= args.num_steps-1:
                print ('save model ...')
		
                torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(args.num_steps)+'.pth'))

            if i_iter % args.save_pred_every == 0 and i_iter!=0:
                print ('taking snapshot ...')
                torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(i_iter)+'.pth'))
            #del loss_seg,loss_score_map, loss_contour_map
            #print('EXCEPTION ------------------------------------>')
            continue
        #pred = F.sigmoid(contour_map)
        #pred=pred.to(device)

        contour_map=contour_map.to(device)
        #print("hello")
        #print("update visdom iter",args.update_visdom_iter,args.visualization)

        if i_iter%args.update_visdom_iter==0 and args.visualization:
            #print("before calling",type(contour_map[0]),contour_map[0].shape,score_map[0].shape)
            # plt.imshow(contour_map[0][0].detach().cpu().numpy())
            # plt.show()
            visual_list = [img[0],compressed_ground_truth[0,0,...],contour_map[0],score_map[0],center_line_map[0], variance_map[0]]
            visualization(visual_list )
            #print("calling visualization",i_iter)
        contour_map=contour_map.squeeze(1)

        # center_line_map=center_line_map.squeeze(1)
        compressed_ground_truth=compressed_ground_truth.squeeze(1)
        #print("torch unique",torch.unique(compressed_ground_truth))
        # print("before loss",pred.size(),contour_map.size(),center_line_map.size()  )
        #center_line_map= center_line_map.unsqueeze(1)
        #center_line_map= center_line_map.to(device)
        #print("before invoking loss",contour_map.size(),compressed_ground_truth.size(),score_map.size(),torch.unique(score_map),center_line_map.unsqueeze(1).size())
        #squeeze the train_mask containing the dont care contour regions
        train_mask= train_mask.squeeze(1)
        score_map=score_map.squeeze(1)
        center_line_map= center_line_map.squeeze(1)
        #print("score map size",score_map.size(),score_map.shape)
        loss_contour_map = loss_dice(train_mask,contour_map,compressed_ground_truth)
        loss_score_map=centre_line_dice_loss(compressed_ground_truth,train_mask,score_map,center_line_map)


        # loss_score_map= loss_dice(score_map,center_line_map)
        # loss_score_map=centre_line_loss(compressed_ground_truth,train_mask,score_map,center_line_map)

        contour_loss_tolerance= 1
        score_map_loss_tolerance= 1
        loss_seg= contour_loss_tolerance*loss_contour_map + score_map_loss_tolerance*loss_score_map

        #print("contour_map",contour_map.shape,"compressed_gt",compressed_ground_truth.shape,"center line map",center_line_map.shape,"score_map",score_map.shape,"train_mask",train_mask.shape)
        if not torch.isnan(loss_seg):
            #print("performing backwardpass")
            loss_seg.backward()
            optimizer.step()
        else:
            print("backprop error")
            outF = open("train_ctw_backprop_errors.txt", "a")
            outF.write(str(i_iter))
            outF.write("\n")
            outF.close()
        # loss_seg_value += loss_seg.data.cpu().numpy()/args.iter_size

        if i_iter%10==0 :
            print('exp = {}'.format(args.snapshot_dir))
            # print("loss_seg",loss_seg)
            # print("loss_score_map",loss_score_map)
            # print("loss_contour_map",loss_contour_map)
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
            print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_gaussian_branch = {3:.3f}, loss_segmentation_branch = {4:.3f}'.format(i_iter, args.num_steps, loss_seg,loss_contour_map,loss_score_map))
        del score_map,contour_map, flag,variance_map, image, compressed_ground_truth,center_line_map,train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map
        del loss_seg,loss_score_map, loss_contour_map

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(args.num_steps)+'.pth'))

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(model_save_dir, args.dataset+'_3d_rotated_gaussian_without_attention_'+str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
