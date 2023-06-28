import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import time
import skimage.measure
import torch.nn.functional as F

from PIL import Image
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from multiprocessing import Pool


def findMaxConnectedRegion(x):
    assert len(x.shape) == 2
    cc, num = skimage.measure.label(x, connectivity=1, return_num=True)
    omega = np.zeros_like(x)
    if num > 0:
        # find the largest connected region
        max_id = np.argmax(np.bincount(cc.flatten())[1:]) + 1
        omega[cc == max_id] = 1
    return omega

def genGaussKernel(sigma, q=2):
    pi = math.pi
    eps = 1e-2

    def gauss(x, sigma):
        return np.exp(-np.power(x,2)/(2*np.power(sigma,2))) / (sigma*np.sqrt(2*pi))

    def dgauss(x, sigma):
        return -x * gauss(x,sigma) / np.power(sigma, 2)

    hsize = int(np.ceil(sigma*np.sqrt(-2*np.log(np.sqrt(2*pi)*sigma*eps))))
    size = 2 * hsize + 1
    hx = np.zeros([size, size], dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u, v = i-hsize, j-hsize
            hx[i,j] = gauss(u,sigma) * dgauss(v,sigma)

    hx = hx / np.sqrt(np.sum(np.power(np.abs(hx), 2)))
    hy = hx.transpose(1, 0)
    return hx, hy, size

def calcOpticalFlow(frames):
    prev, curr = frames
    flow = cv2.calcOpticalFlowFarneback(prev.astype(np.uint8), curr.astype(np.uint8), None,
                                        0.5, 5, 10, 2, 7, 1.5,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow


class ImageFilter(nn.Module):
    def __init__(self, chn, kernel_size, weight, device):
        super(ImageFilter, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size == weight.size(-1)
        self.filter = nn.Conv2d(chn, chn, kernel_size, padding=0, bias=False)
        self.filter.weight = nn.Parameter(weight)
        self.device = device

    def pad(self, x):
        assert len(x.shape) == 3
        x = x.unsqueeze(-1).permute((0,3,1,2))
        b, c, h, w = x.shape
        pad = self.kernel_size // 2
        y = torch.zeros([b, c, h+pad*2, w+pad*2]).to(self.device)
        y[:,:,0:pad,0:pad] = x[:,:,0:1,0:1].repeat(1,1,pad,pad)
        y[:,:,0:pad,w+pad:] = x[:,:,0:1,-1:].repeat(1,1,pad,pad)
        y[:,:,h+pad:,0:pad] = x[:,:,-1:,0:1].repeat(1,1,pad,pad)
        y[:,:,h+pad:,w+pad:] = x[:,:,-1:,-1:].repeat(1,1,pad,pad)

        y[:,:,0:pad,pad:w+pad] = x[:,:,0:1,:].repeat(1,1,pad,1)
        y[:,:,pad:h+pad,0:pad] = x[:,:,:,0:1].repeat(1,1,1,pad)
        y[:,:,h+pad:,pad:w+pad] = x[:,:,-1:,:].repeat(1,1,pad,1)
        y[:,:,pad:h+pad,w+pad:] = x[:,:,:,-1:].repeat(1,1,1,pad)

        y[:,:,pad:h+pad, pad:w+pad] = x
        return y

    def forward(self, x):
        y = self.filter(self.pad(x))
        return y


class BatchMetric(object):
    def __init__(self, device, grad_sigma=1.4, grad_q=2,
                       conn_step=0.1, conn_thresh=0.5, conn_theta=0.15, conn_p=1):
        # parameters for connectivity
        self.conn_step = conn_step
        self.conn_thresh = conn_thresh
        self.conn_theta = conn_theta
        self.conn_p = conn_p
        self.device = device

        hx, hy, size = genGaussKernel(grad_sigma, grad_q)
        self.hx = hx
        self.hy = hy
        self.kernel_size = size
        kx = self.hx[::-1, ::-1].copy()
        ky = self.hy[::-1, ::-1].copy()
        kernel_x = torch.from_numpy(kx).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.from_numpy(ky).unsqueeze(0).unsqueeze(0)
        self.fx = ImageFilter(1, self.kernel_size, kernel_x, self.device).cuda(self.device)
        self.fy = ImageFilter(1, self.kernel_size, kernel_y, self.device).cuda(self.device)

    def run(self, input, target, mask=None, calc_mad=False):
        torch.cuda.empty_cache()
        input_t = torch.from_numpy(input.astype(np.float32)).to(self.device)
        target_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        if mask is None:
            mask = torch.ones_like(target_t).to(self.device)
        else:
            mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            mask = (mask == 128).float()
        if calc_mad:
            mad = self.BatchMAD(input_t, target_t, mask)
        else:
            mad = None
        sad = self.BatchSAD(input_t, target_t, mask)
        mse = self.BatchMSE(input_t, target_t, mask)
        grad = self.BatchGradient(input_t, target_t, mask)
        conn = self.BatchConnectivity(input_t, target_t, mask)
        return sad, mad, mse, grad, conn

    def run_quick(self, input, target, mask=None):
        torch.cuda.empty_cache()
        input_t = torch.from_numpy(input.astype(np.float32)).to(self.device)
        target_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        if mask is None:
            mask = torch.ones_like(target_t).to(self.device)
        else:
            mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            mask = (mask == 128).float()
            
        mad = self.BatchMAD(input_t, target_t, mask)
        #sad = self.BatchSAD(input_t, target_t, mask)
        mse = self.BatchMSE(input_t, target_t, mask)
        #grad = self.BatchGradient(input_t, target_t, mask)
        #conn = self.BatchConnectivity(input_t, target_t, mask)
        return mad, mse

    def run_metric(self, metric, input, target, mask=None):
        torch.cuda.empty_cache()
        input_t = torch.from_numpy(input.astype(np.float32)).to(self.device)
        target_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        if mask is None:
            mask = torch.ones_like(target_t).to(self.device)
        else:
            mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            mask = (mask == 128).float()

        if metric == 'sad':
            ret = self.BatchSAD(input_t, target_t, mask)
        elif metric == 'mse':
            ret = self.BatchMSE(input_t, target_t, mask)
        elif metric == 'grad':
            ret = self.BatchGradient(input_t, target_t, mask)
        elif metric == 'conn':
            ret = self.BatchConnectivity(input_t, target_t, mask)
        else:
            raise NotImplementedError
        return ret

    def BatchSAD(self, pred, target, mask):
        B = target.size(0)
        error_map = (pred - target).abs() / 255.
        batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
        batch_loss = batch_loss / 1000.
        return batch_loss.data.cpu().numpy()

    def BatchMAD(self, pred, target, mask):
        B = target.size(0)
        error_map = (pred - target).abs() / 255.
        batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
        batch_loss = batch_loss / (mask.view(B, -1).sum(dim=-1) + 1.)
        return batch_loss.data.cpu().numpy()

    def BatchMSE(self, pred, target, mask):
        B = target.size(0)
        error_map = (pred-target) / 255.
        batch_loss = (error_map.pow(2) * mask).view(B, -1).sum(dim=-1)
        batch_loss = batch_loss / (mask.view(B, -1).sum(dim=-1) + 1.)
        return batch_loss.data.cpu().numpy()

    def BatchGradient(self, pred, target, mask):
        B = target.size(0)
        pred = pred / 255.
        target = target / 255.

        pred_x_t = self.fx(pred).squeeze(1)
        pred_y_t = self.fy(pred).squeeze(1)
        target_x_t = self.fx(target).squeeze(1)
        target_y_t = self.fy(target).squeeze(1)
        pred_amp = (pred_x_t.pow(2) + pred_y_t.pow(2)).sqrt()
        target_amp = (target_x_t.pow(2) + target_y_t.pow(2)).sqrt()
        error_map = (pred_amp - target_amp).pow(2)
        batch_loss = (error_map * mask).view(B, -1).sum(dim=-1) / (mask.view(B,-1).sum(dim=-1) + 1.)
        return batch_loss.data.cpu().numpy()

    def BatchConnectivity(self, pred, target, mask):
        _, h, w = pred.shape

        step = self.conn_step
        theta = self.conn_theta

        pred = pred / 255.
        target = target / 255.
        B, dimy, dimx = pred.shape
        thresh_steps = torch.arange(0, 1+step, step).to(self.device)
        l_map = torch.ones_like(pred).to(self.device)*(-1)
        pool = Pool(B)
        for i in range(1, len(thresh_steps)):
            pred_alpha_thresh = pred>=thresh_steps[i]
            target_alpha_thresh = target>=thresh_steps[i]
            mask_i = pred_alpha_thresh * target_alpha_thresh
            omegas = []
            items = [mask_ij.data.cpu().numpy() for mask_ij in mask_i]
            for omega in pool.imap(findMaxConnectedRegion, items):
                omegas.append(omega)
            omegas = torch.from_numpy(np.array(omegas)).to(self.device)
            flag = (l_map==-1) * (omegas==0)
            l_map[flag==1] = thresh_steps[i-1]
        l_map[l_map==-1] = 1
        pred_d = pred - l_map
        target_d = target - l_map
        pred_phi = 1 - pred_d*(pred_d>=theta).float()
        target_phi = 1 -  target_d*(target_d>=theta).float()
        batch_loss = ((pred_phi-target_phi).abs()*mask).view(B, -1).sum(dim=-1) / (mask.view(B,-1).sum(dim=-1) + 1.)
        pool.close()
        return batch_loss.data.cpu().numpy()

    def GaussianGradient(self, mat):
        gx = np.zeros_like(mat)
        gy = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            gx[i, ...] = ndimage.filters.convolve(mat[i], self.hx, mode='nearest')
            gy[i, ...] = ndimage.filters.convolve(mat[i], self.hy, mode='nearest')
        return gx, gy


def generate_trimap(alpha, k_size=3, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=iterations)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)
