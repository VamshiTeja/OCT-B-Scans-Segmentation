# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2019-02-20 03:46:27
# @Last Modified by:   vamshi
# @Last Modified time: 2019-02-20 11:33:35
import numpy as np
from helpers import *
import helpers

def threshold(img):
	b,g,r = cv2.split(img)
	_, b = cv2.threshold(b, 127,255,cv2.THRESH_BINARY)
	_, g = cv2.threshold(g, 127,255,cv2.THRESH_BINARY)
	_, r = cv2.threshold(r, 127,255,cv2.THRESH_BINARY)
	img = cv2.merge((b,g,r))
	return img

n_cl = 3
def mean_Dice(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)
    cl, n_cl   = union_classes(eval_segm, gt_segm)
    cl = [0, 1, 2]
    n_cl = 3
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([int(0)]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) and (np.sum(curr_gt_mask) == 0):
            IU[i] = 'nan'
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        IU[i] = 2*n_ii / (t_i + n_ij)
 	
 	print IU
    # mean_IU_ = np.sum(IU) / n_cl_gt
    return 0, IU


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    n_cl = 3
    cl = [0,1,2]
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = [int(0)] * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) and (np.sum(curr_gt_mask) == 0):
            IU[i] = 'nan'

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue


        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    # mean_IU_ = np.sum(IU) / n_cl_gt
    return 0, IU

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# def compute_mean_iou(pred, label):

#     unique_labels = np.unique(label)
#     num_unique_labels = len(unique_labels);

#     I = np.zeros(num_unique_labels)
#     U = np.zeros(num_unique_labels)

#     for index, val in enumerate(unique_labels):
#         pred_i = pred == val
#         label_i = label == val

#         I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
#         U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


#     mean_iou = np.mean(I / U)
#     return mean_iou, I/U

# def compute_dice_coeff(pred,label):
#     '''
#     pred: predicted image 2D image with depth 1 cotaining class values 
#     gt: ground truth  2D
#     '''

#     unique_labels = np.unique(label)
#     num_unique_labels = len(unique_labels);

#     N = np.zeros(num_unique_labels)
#     D = np.zeros(num_unique_labels)

#     for index, val in enumerate(unique_labels):
#         pred_i = pred == val
#         label_i = label == val

#         N[index] = 2.0*float(np.sum(np.logical_and(label_i, pred_i)))
#         D[index] = float(np.sum(label_i))+float(np.sum(pred_i))

#     mean_dice = np.mean(N / D)
#     return mean_dice, N/D


########### UNet ######################

# class_names_list, label_values = helpers.get_label_info(os.path.join("SRF_PED", "class_dict.csv"))

# UNet_DIR = "unet_test_Val/"

# dice = []
# iou = []
# images = sorted(os.listdir(UNet_DIR))
# for i in range(1,len(images),3):
# 	gt = helpers.reverse_one_hot(helpers.one_hot_it(cv2.imread(os.path.join(UNet_DIR, images[i])), label_values))
# 	pred = helpers.reverse_one_hot(helpers.one_hot_it(cv2.imread(os.path.join(UNet_DIR, images[i+1])), label_values)) 

# 	#_, d = compute_dice_coeff(pred, gt)
# 	#_, i = compute_mean_iou(pred, gt)

# 	_, i = mean_IU(pred, gt)
# 	_, d = mean_Dice(pred, gt)

# 	dice.append(d)
# 	iou.append(i)


# dice = np.array(dice)
# iou = np.array(iou)	

# SRF_d = dice[:,0]
# SRF_dice = [i for i in SRF_d if i>0]

# PED_d = dice[:,1]
# PED_dice = [i for i in PED_d if i>0]


# SRF_i = iou[:,0]
# SRF_iou = [i for i in SRF_i if i>0]

# PED_i = iou[:,1]
# PED_iou = [i for i in PED_i if i>0]


# print np.mean(SRF_dice), np.mean(PED_dice)
# print np.mean(SRF_iou), np.mean(PED_iou)



############ DeepLab v3 ##########################

class_names_list, label_values = helpers.get_label_info(os.path.join("SRF_PED", "class_dict.csv"))
DeepLab_DIR = "deeplab_test_Val"

dice = []
iou = []
images = sorted(os.listdir(DeepLab_DIR))
for i in range(0, len(images)-1,2):
	gt_image = threshold(cv2.imread(os.path.join(DeepLab_DIR, images[i])))
	pred_image = threshold(cv2.imread(os.path.join(DeepLab_DIR, images[i+1])))
	
	# print(helpers.one_hot_it(pred_image, label_values).shape)

	gt = helpers.reverse_one_hot(helpers.one_hot_it(gt_image, label_values))
	pred = helpers.reverse_one_hot(helpers.one_hot_it(pred_image, label_values)) 
	# print np.unique(pred)
	#_, d = compute_dice_coeff(pred, gt)
	#_, i = compute_mean_iou(pred, gt)

	_, i = mean_IU(pred, gt)
	_, d = mean_Dice(pred, gt)

	dice.append(d)
	iou.append(i)


dice = np.array(dice)
iou = np.array(iou)	


SRF_d = dice[:,0]
SRF_dice = [float(i) for i in SRF_d if i>0 and i!='nan']

PED_d = dice[:,1]
PED_dice = [float(i) for i in PED_d if i>0 and i!='nan']


SRF_i = iou[:,0]
SRF_iou = [float(i) for i in SRF_i if i>0 and i!='nan']

PED_i = iou[:,1]
PED_iou = [float(i) for i in PED_i if i>0 and i!='nan']

print PED_iou
print SRF_iou

print np.mean(SRF_dice), np.mean(PED_dice)
print np.mean(SRF_iou), np.mean(PED_iou)