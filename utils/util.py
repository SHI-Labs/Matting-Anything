import os
import cv2
import torch
import logging
import numpy as np
from utils.config import CONFIG
import torch.distributed as dist
import torch.nn.functional as F
from skimage.measure import label
import pdb

def make_dir(target_dir):
    """
    Create dir if not exists
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def print_network(model, name):
    """
    Print out the network information
    """
    logger = logging.getLogger("Logger")
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    logger.info(model)
    logger.info(name)
    logger.info("Number of parameters: {}".format(num_params))


def update_lr(lr, optimizer):
    """
    update learning rates
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(init_lr, step, iter_num):
    """
    Warm up learning rate
    """
    return step/iter_num*init_lr


def add_prefix_state_dict(state_dict, prefix="module"):
    """
    add prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[prefix+"."+key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    return new_state_dict


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict


def load_imagenet_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
    state_dict = remove_prefix_state_dict(checkpoint['state_dict'])
    for key, value in state_dict.items():
        state_dict[key] = state_dict[key].float()

    logger = logging.getLogger("Logger")
    logger.debug("Imagenet pretrained keys:")
    logger.debug(state_dict.keys())
    logger.debug("Generator keys:")
    logger.debug(model.module.encoder.state_dict().keys())
    logger.debug("Intersection  keys:")
    logger.debug(set(model.module.encoder.state_dict().keys())&set(state_dict.keys()))

    weight_u = state_dict["conv1.module.weight_u"]
    weight_v = state_dict["conv1.module.weight_v"]
    weight_bar = state_dict["conv1.module.weight_bar"]

    logger.debug("weight_v: {}".format(weight_v))
    logger.debug("weight_bar: {}".format(weight_bar.view(32, -1)))
    logger.debug("sigma: {}".format(weight_u.dot(weight_bar.view(32, -1).mv(weight_v))))

    new_weight_v = torch.zeros((3+CONFIG.model.mask_channel), 3, 3).cuda()
    new_weight_bar = torch.zeros(32, (3+CONFIG.model.mask_channel), 3, 3).cuda()

    new_weight_v[:3, :, :].copy_(weight_v.view(3, 3, 3))
    new_weight_bar[:, :3, :, :].copy_(weight_bar)

    logger.debug("new weight_v: {}".format(new_weight_v.view(-1)))
    logger.debug("new weight_bar: {}".format(new_weight_bar.view(32, -1)))
    logger.debug("new sigma: {}".format(weight_u.dot(new_weight_bar.view(32, -1).mv(new_weight_v.view(-1)))))

    state_dict["conv1.module.weight_v"] = new_weight_v.view(-1)
    state_dict["conv1.module.weight_bar"] = new_weight_bar

    model.module.encoder.load_state_dict(state_dict, strict=False)

def load_imagenet_pretrain_nomask(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
    state_dict = remove_prefix_state_dict(checkpoint['state_dict'])
    for key, value in state_dict.items():
        state_dict[key] = state_dict[key].float()

    logger = logging.getLogger("Logger")
    logger.debug("Imagenet pretrained keys:")
    logger.debug(state_dict.keys())
    logger.debug("Generator keys:")
    logger.debug(model.module.encoder.state_dict().keys())
    logger.debug("Intersection  keys:")
    logger.debug(set(model.module.encoder.state_dict().keys())&set(state_dict.keys()))

    #weight_u = state_dict["conv1.module.weight_u"]
    #weight_v = state_dict["conv1.module.weight_v"]
    #weight_bar = state_dict["conv1.module.weight_bar"]

    #logger.debug("weight_v: {}".format(weight_v))
    #logger.debug("weight_bar: {}".format(weight_bar.view(32, -1)))
    #logger.debug("sigma: {}".format(weight_u.dot(weight_bar.view(32, -1).mv(weight_v))))

    #new_weight_v = torch.zeros((3+CONFIG.model.mask_channel), 3, 3).cuda()
    #new_weight_bar = torch.zeros(32, (3+CONFIG.model.mask_channel), 3, 3).cuda()

    #new_weight_v[:3, :, :].copy_(weight_v.view(3, 3, 3))
    #new_weight_bar[:, :3, :, :].copy_(weight_bar)

    #logger.debug("new weight_v: {}".format(new_weight_v.view(-1)))
    #logger.debug("new weight_bar: {}".format(new_weight_bar.view(32, -1)))
    #logger.debug("new sigma: {}".format(weight_u.dot(new_weight_bar.view(32, -1).mv(new_weight_v.view(-1)))))

    #state_dict["conv1.module.weight_v"] = new_weight_v.view(-1)
    #state_dict["conv1.module.weight_bar"] = new_weight_bar

    model.module.encoder.load_state_dict(state_dict, strict=False)

def load_VGG_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location = lambda storage, loc: storage.cuda())
    backbone_state_dict = remove_prefix_state_dict(checkpoint['state_dict'])

    model.module.encoder.load_state_dict(backbone_state_dict, strict=False)


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if trimap.shape[1] == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()
    return weight

def get_gaborfilter(angles):
    """
    generate gabor filter as the conv kernel
    :param angles: number of different angles
    """
    gabor_filter = []
    for angle in range(angles):
        gabor_filter.append(cv2.getGaborKernel(ksize=(5,5), sigma=0.5, theta=angle*np.pi/8, lambd=5, gamma=0.5))
    gabor_filter = np.array(gabor_filter)
    gabor_filter = np.expand_dims(gabor_filter, axis=1)
    return gabor_filter.astype(np.float32)


def get_gradfilter():
    """
    generate gradient filter as the conv kernel
    """
    grad_filter = []
    grad_filter.append([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_filter.append([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_filter = np.array(grad_filter)
    grad_filter = np.expand_dims(grad_filter, axis=1)
    return grad_filter.astype(np.float32)


def reduce_tensor_dict(tensor_dict, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, mode)
    return tensor_dict


def reduce_tensor(tensor, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= CONFIG.world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt

### preprocess the image and mask for inference (np array), crop based on ROI
def preprocess(image, mask, thres):
    mask_ = (mask >= thres).astype(np.float32)
    arr = np.nonzero(mask_)
    h, w = mask.shape
    bbox = [max(0, int(min(arr[0]) - 0.1*h)),
            min(h, int(max(arr[0]) + 0.1*h)),
            max(0, int(min(arr[1]) - 0.1*w)),
            min(w, int(max(arr[1]) + 0.1*w))]
    image = image[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return image, mask, bbox

### postprocess the alpha prediction to keep the largest connected component (np array) and uncrop, alpha in [0, 1]
### based on https://github.com/senguptaumd/Background-Matting/blob/master/test_background-matting_image.py
def postprocess(alpha, orih=None, oriw=None, bbox=None):
    labels=label((alpha>0.05).astype(int))
    try:
        assert( labels.max() != 0 )
    except:
        return None
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    alpha = alpha * largestCC
    if bbox is None:
        return alpha
    else:
        ori_alpha = np.zeros(shape=[orih, oriw], dtype=np.float32)
        ori_alpha[bbox[0]:bbox[1], bbox[2]:bbox[3]] = alpha
        return ori_alpha


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape

    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0
    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_
    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).cuda()
    
    return weight

def get_unknown_tensor_from_pred_oneside(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    #uncertain_area[pred>1-1.0/255.0] = 0
    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_
    uncertain_area[pred>1-1.0/255.0] = 0
    #weight = np.zeros_like(uncertain_area)
    #weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(uncertain_area).cuda()
    return weight

Kernels_mask = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_mask(mask, rand_width=30, train_mode=True):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    N, C, H, W = mask.shape
    mask_c = mask.data.cpu().numpy().astype(np.uint8)

    weight = np.ones_like(mask_c, dtype=np.uint8)

    for n in range(N):
        if train_mode:
            width = np.random.randint(rand_width // 2, rand_width)
        else:
            width = rand_width // 2
        fg_mask = cv2.erode(mask_c[n,0], Kernels_mask[width])
        bg_mask = cv2.erode(1 - mask_c[n,0], Kernels_mask[width])
        weight[n,0][fg_mask==1] = 0
        weight[n,0][bg_mask==1] = 0
    weight = torch.from_numpy(weight).cuda()
    return weight

def get_unknown_tensor_from_mask_oneside(mask, rand_width=30, train_mode=True):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    N, C, H, W = mask.shape
    mask_c = mask.data.cpu().numpy().astype(np.uint8)

    weight = np.ones_like(mask_c, dtype=np.uint8)

    for n in range(N):
        if train_mode:
            width = np.random.randint(rand_width // 2, rand_width)
        else:
            width = rand_width // 2
        #fg_mask = cv2.erode(mask_c[n,0], Kernels_mask[width])
        fg_mask = mask_c[n,0]
        bg_mask = cv2.erode(1 - mask_c[n,0], Kernels_mask[width])
        weight[n,0][fg_mask==1] = 0
        weight[n,0][bg_mask==1] = 0
    weight = torch.from_numpy(weight).cuda()
    return weight

def get_unknown_box_from_mask(mask):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    N, C, H, W = mask.shape
    mask_c = mask.data.cpu().numpy().astype(np.uint8)

    weight = np.ones_like(mask_c, dtype=np.uint8)
    fg_set = np.where(mask_c[0][0] != 0)
    x_min = np.min(fg_set[1])
    x_max = np.max(fg_set[1])
    y_min = np.min(fg_set[0])
    y_max = np.max(fg_set[0])

    weight[0, 0, y_min:y_max, x_min:x_max] = 0
    weight = torch.from_numpy(weight).cuda()
    return weight