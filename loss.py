import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def MSELoss(prediction, target):
    mse = torch.mean((prediction-target)**2)
    return mse

def loss_mse(variance_map,compressed_ground_truth, train_mask):
    # manage the size issue
    variance_map = variance_map.squeeze()
    compressed_ground_truth = compressed_ground_truth.squeeze()
    train_mask = train_mask.squeeze()
    if len(variance_map.shape)==3:
        variance_map = variance_map.unsqueeze(0)
        compressed_ground_truth = compressed_ground_truth.unsqueeze(0)
        train_mask = train_mask.unsqueeze(0)


    variance_x =variance_map[:,0,...]
    # print('hello',variance_x.shape, variance_map.shape)
    variance_y = variance_map[:,1,...]
    # print('hello y', variance_y)
    # print("size",variance_map.size(),\
    #     variance_x.size(),\
    #     variance_y.size(),\
    #     compressed_ground_truth.size(),\
    #     train_mask.size())
    train_mask = torch.gt(compressed_ground_truth+(1-train_mask*1.0),0)*1.0
    #mask in non text region are 1
    train_mask = 1- train_mask
    # print(train_mask)
    train_mask= torch.gt(train_mask,0)
    # print(train_mask)
    gt = torch.masked_select(compressed_ground_truth, train_mask)
    #print(torch.unique(gt))
    variance_x = torch.masked_select(variance_x, train_mask)
    variance_y = torch.masked_select(variance_y, train_mask)
    # print('variance x ', variance_x.shape, 'gt', gt.shape)
    loss_variance_x =  MSELoss(F.relu(variance_x), gt)
    loss_variance_y = MSELoss(F.relu(variance_y), gt)
    loss_variance = (loss_variance_x+loss_variance_y)/2
    return loss_variance

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



class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target, mask):
        #print(output.shape, target.shape, mask.shape)
        output = torch.masked_select(output.squeeze(),mask)
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = torch.masked_select(target, mask)
        #print('after masking', output.shape, target.shape)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss
