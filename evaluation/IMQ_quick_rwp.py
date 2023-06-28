import numpy as np
import os, sys, tqdm, cv2
from scipy.optimize import linear_sum_assignment
from metrics import BatchMetric

def match(pred, gt):
    # pred: (n,h,w)
    # gt: (m,h,w)
    n, h, w = pred.shape
    m, h, w = gt.shape
    pred_mask = (pred>0)
    gt_mask = (gt>0)
    # (n,w)
    union = np.logical_or(pred_mask[:,None,:,:], gt_mask[None,:,:,:]).sum(axis=(2,3))
    inter = np.logical_and(pred_mask[:,None,:,:], gt_mask[None,:,:,:]).sum(axis=(2,3))
    iou = inter / (union + 1e-8)
    # matched_idx = np.argmax(iou, axis=0) # m
    # matched_iou = np.max(iou, axis=0) # m
    # return matched_idx, matched_iou
    return iou


def mad(pred, gt):
    pred_mask = (pred>0)
    gt_mask = (gt>0)
    union_mask = np.logical_or(pred_mask, gt_mask)
    error = np.abs(pred-gt) * union_mask.astype(np.float32)
    error = error.sum(axis=(1,2)) / (union_mask.sum(axis=(1,2)) + 1.)
    score = 1 - np.minimum(error,1)
    return score


def similarity(pred, gt):
    metric = BatchMetric('cuda')
    # mask = np.logical_or(pred>0, gt>0) * 128
    mask = np.logical_or(pred>0, gt>0) * 128
    mad, mse = metric.run_quick(pred*255, gt*255, mask=mask)
    mad = 1 - np.minimum(mad * 10, 1.0)
    mse = 1 - np.minimum(mse * 10, 1.0)
    #grad = 1 - np.minimum(grad * 10, 1.0)
    #conn = 1 - np.minimum(conn * 10, 1.0)
    score = [mad.sum(), mse.sum()]
    return score


def compute_stats_per_image(pred, gt, thresh_list, func=mad):
    # matched_idx, matched_iou = match(pred, gt)
    tp_list, fp_list, fn_list = [], [], []
    MQ_list = []
    if len(pred)>0 and len(gt)>0:
        iou_matrix = match(pred, gt)
        matched_i, matched_j  = linear_sum_assignment(1-iou_matrix)
        matched_iou = iou_matrix[matched_i, matched_j]
        # score = func(pred[matched_i], gt[matched_j]) * matched_iou
        for thresh in thresh_list:
            tp = (matched_iou>=thresh).sum()
            fp = pred.shape[0] - tp
            fn = gt.shape[0] - tp
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            if tp>0:
                tp_idx = np.where(matched_iou>=thresh)
                tp_i, tp_j = matched_i[tp_idx], matched_j[tp_idx]
                score_list = similarity(pred[tp_i], gt[tp_j])
                #score_list.append(matched_iou[tp_idx].sum())
                MQ_list.append(score_list)
            else:
                MQ_list.append([0,0])
    elif len(pred) == 0:
        for thresh in thresh_list:
            tp_list.append(0)
            fp_list.append(0)
            fn_list.append(len(gt))
            MQ_list.append([0,0])
    else:
        for thresh in thresh_list:
            tp_list.append(0)
            fp_list.append(len(pred))
            fn_list.append(0)
            MQ_list.append([0,0])
    return tp_list, fp_list, fn_list, MQ_list


def compute_stats(pred_folder, gt_folder, thresh_list):
    n_thresh = len(thresh_list)
    IMQ_list = [] # n_thresh, n_IMQ, n_instances
    _MQ_list = []
    _RQ_list = []
    for i in range(n_thresh):
        IMQ_list.append([0]*2)
        _MQ_list.append([0]*2)
        _RQ_list.append([0]*2)
    TP, FP, FN = [0]*n_thresh, [0]*n_thresh, [0]*n_thresh
    for item in tqdm.tqdm(sorted(os.listdir(gt_folder))):
        #if not os.path.exists(os.path.join(pred_folder, item)):
        #    continue
        #pred_images = [cv2.imread(os.path.join(pred_folder, item, im), 0)/255. for im in os.listdir(os.path.join(pred_folder, item))]
        #gt_images = [cv2.imread(os.path.join(gt_folder, item, im), 0)/255. for im in os.listdir(os.path.join(gt_folder, item))]
        pred_images = [cv2.imread(os.path.join(pred_folder, item), 0)/255.]
        gt_images = [cv2.imread(os.path.join(gt_folder, item), 0)/255.]
        if len(pred_images)>0:
            pred_items = np.stack(pred_images, axis=0)
        else:
            pred_items = []
        if len(gt_images)>0:
            gt_items = np.stack(gt_images, axis=0)
        else:
            gt_items = []

        tp_list, fp_list, fn_list, MQ_list = compute_stats_per_image(pred_items, gt_items, thresh_list)
        for i in range(0, n_thresh):
            TP[i] += tp_list[i]
            FP[i] += fp_list[i]
            FN[i] += fn_list[i]
            for j in range(len(MQ_list[i])):
                IMQ_list[i][j] += MQ_list[i][j]
                _MQ_list[i][j] += MQ_list[i][j]
    for i in range(0, n_thresh):
        coeff = 1.0 / (TP[i] + 0.5*FP[i] + 0.5*FN[i] + 1e-6)
        for j in range(len(IMQ_list[0])):
            IMQ_list[i][j] = IMQ_list[i][j] * coeff
            _MQ_list[i][j] = _MQ_list[i][j] / float(TP[i])
            _RQ_list[i][j] = coeff * TP[i]
    return IMQ_list, _MQ_list, _RQ_list


if __name__ == "__main__":
    pred_folder = sys.argv[1]
    gt_folder = sys.argv[2]
    #thresh_list = [0.5, 0.75]
    thresh_list = [0.5]
    IMQ_list, MQ_list, RQ_list = compute_stats(pred_folder, gt_folder, thresh_list)
    for IMQ, MQ, RQ, thresh in zip(IMQ_list, MQ_list, RQ_list, thresh_list):
        print("IMQ/MQ/RQ on Threshold = {}".format(thresh))
        for i, name in enumerate(['MAD', 'MSE']):
            print('{} = {}/{}/{}'.format(name, IMQ[i], MQ[i], RQ[i]))
