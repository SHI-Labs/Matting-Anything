import os
import cv2
import toml
import argparse
import numpy as np
import json

import torch
from torch.nn import functional as F
import torchvision

import utils
from   utils import CONFIG
import networks
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
import sys

sys.path.insert(0, './segment-anything')
sys.path.insert(0, './GroundingDINO')
from segment_anything.utils.transforms import ResizeLongestSide
from groundingdino.util.inference import Model

transform = ResizeLongestSide(1024)

def single_ms_inference(model, image_dict, args):

    with torch.no_grad():
        feas, pred, post_mask = model.forward_inference(image_dict)
        if args.sam:
            post_mask = post_mask[0].cpu().numpy() * 255
            return post_mask.transpose(1, 2, 0).astype('uint8')
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred_os8 = alpha_pred_os8[..., : image_dict['pad_shape'][0], : image_dict['pad_shape'][1]]
        alpha_pred_os4 = alpha_pred_os4[..., : image_dict['pad_shape'][0], : image_dict['pad_shape'][1]]
        alpha_pred_os1 = alpha_pred_os1[..., : image_dict['pad_shape'][0], : image_dict['pad_shape'][1]]

        alpha_pred_os8 = F.interpolate(alpha_pred_os8, image_dict['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os4 = F.interpolate(alpha_pred_os4, image_dict['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os1 = F.interpolate(alpha_pred_os1, image_dict['ori_shape'], mode="bilinear", align_corners=False)
        
        if args.maskguide:
            if args.twoside:
                weight_os8 = utils.get_unknown_tensor_from_mask(post_mask, rand_width=args.os8_width, train_mode=False)
            else:
                weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=args.os8_width, train_mode=False)
            post_mask[weight_os8>0] = alpha_pred_os8[weight_os8>0]
            alpha_pred = post_mask.clone().detach()
        else:
            if args.postprocess:
                weight_os8 = utils.get_unknown_box_from_mask(post_mask)
                alpha_pred_os8[weight_os8>0] = post_mask[weight_os8>0]
            alpha_pred = alpha_pred_os8.clone().detach()

        if args.twoside:
            weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=args.os4_width, train_mode=False)
        else:
            weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=args.os4_width, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        
        if args.twoside:
            weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=args.os1_width, train_mode=False)
        else:
            weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=args.os1_width, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
       
        alpha_pred = alpha_pred[0].cpu().numpy() * 255
        return alpha_pred.transpose(1, 2, 0).astype('uint8')

def generator_tensor_dict(image_path, alpha_path, args):
    # read images
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    alpha_single = cv2.imread(alpha_path, 0)
    alpha_single[alpha_single>127] = 255
    alpha_single[alpha_single<=127] = 0

    fg_set = np.where(alpha_single != 0)
    x_min = np.min(fg_set[1])
    x_max = np.max(fg_set[1])
    y_min = np.min(fg_set[0])
    y_max = np.max(fg_set[0])
    bbox = np.array([x_min, y_min, x_max, y_max])
    
    image = transform.apply_image(image)
    image = torch.as_tensor(image).cuda()
    image = image.permute(2, 0, 1).contiguous()
    bbox = transform.apply_boxes(bbox, original_size)
    input_point = np.array([[(bbox[0][0] + bbox[0][2])/2, (bbox[0][1] + bbox[0][3])/2]])
    input_label = np.array([1])
    input_point = torch.as_tensor(input_point, dtype=torch.float).cuda()
    input_label = torch.as_tensor(input_label, dtype=torch.float).cuda()
    bbox = torch.as_tensor(bbox, dtype=torch.float).cuda()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).cuda()
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).cuda()

    image = (image - pixel_mean) / pixel_std

    h, w = image.shape[-2:]
    pad_size = image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image = F.pad(image, (0, padw, 0, padh))

    if args.prompt == 'box':
        sample = {'image': image[None, ...], 'bbox': bbox[None, ...], 'ori_shape': original_size, 'pad_shape': pad_size}
    elif args.prompt == 'point':
        sample = {'image': image[None, ...], 'point': input_point[None, ...], 'label': input_label[None, ...], 'ori_shape': original_size, 'pad_shape': pad_size}
    return sample


def generator_tensor_dict_from_text(image_path, text, dino_model, args):
    # read images
    image = cv2.imread(image_path)
    detections, phrases = dino_model.predict_with_caption(
        image=image,
        caption=text,
        box_threshold=0.25,
        text_threshold=0.5
        )
    
    if len(detections.xyxy) > 1:
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            0.8,
            ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
    
    bbox = detections.xyxy[np.argmax(detections.confidence)]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    image = transform.apply_image(image)
    image = torch.as_tensor(image).cuda()
    image = image.permute(2, 0, 1).contiguous()
    bbox = transform.apply_boxes(bbox, original_size)
    bbox = torch.as_tensor(bbox, dtype=torch.float).cuda()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).cuda()
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).cuda()

    image = (image - pixel_mean) / pixel_std

    h, w = image.shape[-2:]
    pad_size = image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image = F.pad(image, (0, padw, 0, padh))

    sample = {'image': image[None, ...], 'bbox': bbox[None, ...], 'ori_shape': original_size, 'pad_shape': pad_size}
    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/MAM-ViTB-8gpu.toml')
    parser.add_argument('--benchmark', type=str, default='him2k', choices=['him2k', 'him2k_comp', 'rwp636', 'ppm100', 'am2k', 'pm10k', 'rw100'])
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mam_sam_vitb.pth',
                        help="path of checkpoint")
    parser.add_argument('--image-ext', type=str, default='.jpg', help="input image ext")
    parser.add_argument('--mask-ext', type=str, default='.png', help="input mask ext")
    parser.add_argument('--output', type=str, default='outputs/', help="output dir")
    parser.add_argument('--os8_width', type=int, default=10, help="guidance threshold")
    parser.add_argument('--os4_width', type=int, default=20, help="guidance threshold")
    parser.add_argument('--os1_width', type=int, default=10, help="guidance threshold")
    parser.add_argument('--twoside', action='store_true', default=False, help='post process with twoside of the guidance')        
    parser.add_argument('--sam', action='store_true', default=False, help='return mask')    
    parser.add_argument('--maskguide', action='store_true', default=False, help='mask guidance')    
    parser.add_argument('--postprocess', action='store_true', default=False, help='postprocess to remove bg')    
    parser.add_argument('--prompt', type=str, default='box', choices=['box', 'point', 'text'])

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")
    args.output = os.path.join(args.output)
    utils.make_dir(args.output)

    # build model
    model = networks.get_generator_m2m(seg=CONFIG.model.arch.seg, m2m=CONFIG.model.arch.m2m)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.m2m.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()
    n_parameters = sum(p.numel() for p in model.m2m.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if args.prompt == 'text':
        GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    if args.benchmark == 'him2k':
        image_dir = CONFIG.benchmark.him2k_img
        alpha_dir = CONFIG.benchmark.him2k_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0])
            output_path = os.path.join(args.output, os.path.splitext(image_name)[0])
            utils.make_dir(output_path)

            for alpha_single_dir in sorted(os.listdir(alpha_path)):
                alpha_single_path = os.path.join(alpha_path, alpha_single_dir)
                image_dict = generator_tensor_dict(image_path, alpha_single_path, args)
                alpha_pred = single_ms_inference(model, image_dict, args)
                cv2.imwrite(os.path.join(output_path, alpha_single_dir), alpha_pred)
            
    elif args.benchmark == 'him2k_comp':
        image_dir = CONFIG.benchmark.him2k_comp_img
        alpha_dir = CONFIG.benchmark.him2k_comp_alpha
    
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0])
            output_path = os.path.join(args.output, os.path.splitext(image_name)[0])
            utils.make_dir(output_path)

            for alpha_single_dir in sorted(os.listdir(alpha_path)):
                alpha_single_path = os.path.join(alpha_path, alpha_single_dir)
                image_dict = generator_tensor_dict(image_path, alpha_single_path, args)
                alpha_pred = single_ms_inference(model, image_dict, args)
                cv2.imwrite(os.path.join(output_path, alpha_single_dir), alpha_pred)

    elif args.benchmark == 'rwp636':
        image_dir = CONFIG.benchmark.rwp636_img
        alpha_dir = CONFIG.benchmark.rwp636_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'ppm100':
        image_dir = CONFIG.benchmark.ppm100_img
        alpha_dir = CONFIG.benchmark.ppm100_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, image_name)
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'am2k':
        image_dir = CONFIG.benchmark.am2k_img
        alpha_dir = CONFIG.benchmark.am2k_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'pm10k':
        image_dir = CONFIG.benchmark.pm10k_img
        alpha_dir = CONFIG.benchmark.pm10k_alpha
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
            image_dict = generator_tensor_dict(image_path, alpha_path, args)
            alpha_pred = single_ms_inference(model, image_dict, args)
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)

    elif args.benchmark == 'rw100':
        image_dir = CONFIG.benchmark.rw100_img
        text_dir = CONFIG.benchmark.rw100_text
        index_dir = CONFIG.benchmark.rw100_index
        alpha_dir = CONFIG.benchmark.rw100_alpha
        if args.prompt == 'text':
            index_data = json.load(open(index_dir, 'r'))
            text_data = json.load(open(text_dir, 'r'))
        for i, image_name in enumerate(tqdm(os.listdir(image_dir))):
            if args.prompt == 'text':
                image_path = os.path.join(image_dir, image_name)
                text_label = text_data[os.path.splitext(image_name)[0]]
                index_label = index_data[text_label['image_name']]
                text = text_label['expressions'][index_label]
                image_dict = generator_tensor_dict_from_text(image_path, text, grounding_dino_model, args)
                alpha_pred = single_ms_inference(model, image_dict, args)
                cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)
            else:
                image_path = os.path.join(image_dir, image_name)
                alpha_path = os.path.join(alpha_dir, os.path.splitext(image_name)[0]+'.png')
                image_dict = generator_tensor_dict(image_path, alpha_path, args)
                alpha_pred = single_ms_inference(model, image_dict, args)
                cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'.png'), alpha_pred)