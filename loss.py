import torch
import torch.nn as nn


def loss_dice(dont_care_mask,probas,true_1_hot,eps=1e-7):
    #print('Gaussian branch max values ====>', torch.max(probas))
    gt_mask= (dont_care_mask==1)
    probas=probas*gt_mask
    true_1_hot=true_1_hot*gt_mask

    intersection=torch.sum(probas*true_1_hot,-1)
    intersection=torch.sum(intersection,-1)
    #intersection=torch.sum(intersection,-1)
    cardinality = torch.sum(probas + true_1_hot,-1)
    cardinality = torch.sum(cardinality,-1)
    #cardinality = torch.sum(cardinality,-1)
    #print("inter", cardinality)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    loss = (1 - dice_loss)
    # print(" ======================================> PATA CHALA !!!f ",loss, cardinality)
    # print(" ======================================= bada admin      ", true_1_hot)
    return loss

def centre_line_dice_loss(original_contour_mask,dont_care_mask,pred_center_line,gt_center_line):

    #gt_mask= (original_contour_mask*dont_care_mask==1)
    gt_mask= (dont_care_mask==1)
    # print(dont_care_mask[0])
    # plt.imshow(dont_care_mask.cpu().numpy()[0])
    # plt.show()
    probas=pred_center_line*gt_mask
    true_1_hot=gt_center_line*gt_mask
    #print('size of the tensor ====>', probas.shape, true_1_hot.shape, torch.sum(true_1_hot), torch.max(probas))
    eps=1e-7

    #intersection=probas*true_1_hot
    #cardinality = probas + true_1_hot,
    #print(intersection.shape, cardinality.shape)
    intersection=torch.sum(probas*true_1_hot,-1)
    intersection=torch.sum(intersection,-1)
    #intersection=torch.sum(intersection,-1)
    cardinality = torch.sum(probas + true_1_hot,-1)
    cardinality = torch.sum(cardinality,-1)
    #cardinality = torch.sum(cardinality,-1)

    #print("inter")
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    loss = (1 - dice_loss)
    #print(" ======================================> PATA CHALA !!!f ",loss, cardinality)
    #print(" ======================================= bada admin      ", true_1_hot)
    #print(" f ",loss)
    return loss

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    pred = pred.squeeze()
    #print('cross check', label.shape, pred.shape)
    #print('check label', torch.unique(label))
    label = Variable(label.float()).cuda(gpu)

    pred = torch.clamp(pred, min=1e-7, max=0.99)
    entropy = -label*torch.log(pred)- (1-label)*torch.log(1-pred)
    entropy = torch.mean(torch.mean(entropy,dim=-1),dim=-1)
    entropy = torch.mean(entropy)
    print(entropy.data)
    #criterion = CrossEntropy2d().cuda(gpu)
    #print("emtered function")
    #return criterion(pred, label)
    return entropy

def centre_line_loss(original_contour_mask,dont_care_mask,pred_center_line,gt_center_line):
    #get the gt_mask
    gt_mask= (original_contour_mask*dont_care_mask==1)
    #return nn.crossentropy(pred_center_line[gt_mask],gt_center_line[gt_mask])
    return binary_cross_entropy(pred_center_line[gt_mask],gt_center_line[gt_mask])
    #label is the

def binary_cross_entropy(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #pred = pred.squeeze()
    #print('cross check', label.shape, pred.shape)
    #print('check label', torch.unique(label))
    #label = Variable(label.float()).cuda(gpu)

    pred = torch.clamp(pred, min=1e-7, max=0.99)
    entropy = -label*torch.log(pred)- (1-label)*torch.log(1-pred)
    entropy = torch.mean(torch.mean(entropy,dim=-1),dim=-1)
    entropy = torch.mean(entropy)
    #print(entropy.data)
    #criterion = CrossEntropy2d().cuda(gpu)
    #print("emtered function")
    #return criterion(pred, label)
    #print("entropy",entropy)
    return entropy
