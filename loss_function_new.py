import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

eps = 1e-6

def loss_dice(dont_care_mask,probas,true_1_hot,eps=1e-7):
    gt_mask= (dont_care_mask==1)
    probas=probas*gt_mask
    true_1_hot=true_1_hot*gt_mask

    intersection=torch.sum(probas*true_1_hot,-1)
    intersection=torch.sum(intersection,-1)

    cardinality = torch.sum(probas + true_1_hot,-1)
    cardinality = torch.sum(cardinality,-1)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    loss = (1 - dice_loss)
    # print(" ======================================> PATA CHALA !!!f ",loss, cardinality)
    # print(" ======================================= bada admin      ", true_1_hot)
    return loss

def centre_line_loss(dont_care_mask,pred_center_line,gt_center_line):

    # make the dimensions of the tensor to be the same 
    dont_care_mask = dont_care_mask.squeeze()
    pred_center_line = pred_center_line.squeeze()
    gt_center_line = gt_center_line.squeeze()

    if len(pred_center_line.shape)==3:
        pred_center_line = pred_center_line.unsqueeze(1)
        dont_care_mask = dont_care_mask.unsqueeze(1)
        gt_center_line = gt_center_line.unsqueeze(1)

    if len(pred_center_line.shape)==2:
        pred_center_line = pred_center_line.unsqueeze(0).unsqueeze(0)
        dont_care_mask = dont_care_mask.unsqueeze(0).unsqueeze(0)
        gt_center_line = gt_center_line.unsqueeze(0).unsqueeze(0)

    # perfrom upsampling to reach to the same scale of the prediction 
    pred_center_line = F.interpolate(pred_center_line, mode='nearest', scale_factor=4)
    dont_care_mask=   F.interpolate(dont_care_mask*1.0, mode='nearest', scale_factor=4)

    # find the weight vector for the dice loss 
    bce_map = F.binary_cross_entropy(pred_center_line, gt_center_line, reduction='none')[:, 0, :, :]
    bce_map = (bce_map - bce_map.min()) / (bce_map.max() - bce_map.min()) +1


    gt_mask= (dont_care_mask==1)
    probas=pred_center_line*gt_mask
    true_1_hot=gt_center_line*gt_mask
    eps=1e-7

    intersection=torch.sum(probas*true_1_hot*bce_map,-1)
    intersection=torch.sum(intersection,-1)
    cardinality = torch.sum(probas*bce_map + true_1_hot*bce_map,-1)
    cardinality = torch.sum(cardinality,-1)

    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    loss = (1 - dice_loss)
    #print(" ======================================> PATA CHALA !!!f ",loss, cardinality)
    #print(" ======================================= bada admin      ", true_1_hot)
    #print(" f ",loss)
    return loss


def dice_loss(pred, gt, m):
    intersection = torch.sum(pred*gt*m)+eps
    union = torch.sum(pred*m) + torch.sum(gt*m) + eps
    loss = 1 - 2.0*intersection/union
    if loss > 1 or loss<0:
        print(2.0*intersection, union)
    return loss

def dice_ohnm_loss(pred, gt, m):
    pos_index = (gt == 1) * (m == 1)
    neg_index = (gt == 0) * (m == 1)
    pos_num = pos_index.float().sum().item()
    neg_num = neg_index.float().sum().item()
    if pos_num == 0 or neg_num < pos_num*3.0:
        return dice_loss(pred, gt, m)
    else:
        neg_num = int(pos_num*3)
        pos_pred = pred[pos_index]
        neg_pred = pred[neg_index]
        neg_sort, _ = torch.sort(neg_pred, descending=True)
        sampled_neg_pred = neg_sort[:neg_num]
        pos_gt = pos_pred.clone()
        pos_gt.data.fill_(1.0)
        pos_gt = pos_gt.detach()
        neg_gt = sampled_neg_pred.clone()
        neg_gt.data.fill_(0)
        neg_gt = neg_gt.detach()
        tpred = torch.cat((pos_pred, sampled_neg_pred))
        tgt = torch.cat((pos_gt, neg_gt))
        intersection = torch.sum(tpred * tgt) + eps
        union = torch.sum(tpred) + torch.sum(gt) + eps
        loss = 1 - 2.0 * intersection / union
        return loss

def centre_line_loss2(dont_care_mask, score_map, gt_center_line, textregion_orignal_scale):
    #print('check', dont_care_mask.shape, score_map.shape, gt_center_line.shape, textregion_orignal_scale.shape)
    # make the dimensions of the tensor to be the same 
    dont_care_mask = dont_care_mask.squeeze()
    score_map = score_map.squeeze()
    gt_center_line = gt_center_line.squeeze()

    if len(score_map.shape)==3:
        score_map = score_map.unsqueeze(0)
        dont_care_mask = dont_care_mask.unsqueeze(0).unsqueeze(0)
        gt_center_line = gt_center_line.unsqueeze(0).unsqueeze(0)
    else:
        dont_care_mask = dont_care_mask.unsqueeze(1)
        gt_center_line = gt_center_line.unsqueeze(1)
 

    # perfrom upsampling to reach to the same scale of the prediction 
    score_map = F.interpolate(score_map, mode='nearest', scale_factor=4)
    dont_care_mask=   F.interpolate(dont_care_mask*1.0, mode='nearest', scale_factor=4)

    prediction_center_line = F.softmax(score_map[:,0:2,...],dim=1)
    text_nontext_prediction = F.sigmoid(score_map[:,2,...])
    #gt_mask= (dont_care_mask==1) 
    #print('unique value', torch.unique(dont_care_mask))
    loss1 = text_nontext_loss(dont_care_mask, text_nontext_prediction, textregion_orignal_scale)
    loss2 = center_region_loss_function(dont_care_mask, prediction_center_line, gt_center_line, textregion_orignal_scale)
    loss = loss1 + loss2 
    return loss 	
import matplotlib.pyplot as plt

def text_nontext_loss(gt_mask, text_nontext_prediction, textregion_orignal_scale):
    #print('shape', gt_mask.shape, text_nontext_prediction.shape, textregion_orignal_scale.shape)
    #plt.imshow(textregion_orignal_scale[0].cpu().numpy())
    #plt.show()
    loss= dice_ohnm_loss(text_nontext_prediction, textregion_orignal_scale, gt_mask.squeeze())
    #print('text nontextloss ', loss.item())
    return loss 

    
def center_region_loss_function(gt_mask, prediction_center_line, gt_center_line, textregion_orignal_scale):
    lambda1=1.0
    lambda2=0.5
    gt_mask = gt_mask.squeeze()
    mask = gt_mask*textregion_orignal_scale
    pred  = prediction_center_line[:,0,...]

    text_region_center_line_loss  = dice_loss(pred, gt_center_line, mask)

    mask = gt_mask*(1-textregion_orignal_scale) 
    pred  = prediction_center_line[:,1,...]

    #print('unique value', torch.unique(mask))
    #print('gt center line unique value',torch.unique(1-gt_center_line))
    #print('prediction unique value', torch.min(pred))
    nontext_region_center_line_loss  = dice_loss(pred, 1-textregion_orignal_scale, mask)
    loss = lambda1 *text_region_center_line_loss + lambda2*nontext_region_center_line_loss
    #print('center line loses', text_region_center_line_loss.item(), nontext_region_center_line_loss.item())
    return loss 
