from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from fbpn_sr_rbpn_v1 import Net as FBPNSR_RBPN_V1
from fbpn_sr_rbpn_v2 import Net as FBPNSR_RBPN_V2
from fbpn_sr_rbpn_v3 import Net as FBPNSR_RBPN_V3
from fbpn_sr_rbpn_v4 import Net as FBPNSR_RBPN_V4
from fbpn_sr_rbpn_v1_ref import Net as FBPNSR_RBPN_V1_REF
from fbpn_sr_rbpn_v2_ref import Net as FBPNSR_RBPN_V2_REF
from fbpn_sr_rbpn_v3_ref import Net as FBPNSR_RBPN_V3_REF
from fbpn_sr_rbpn_v4_ref import Net as FBPNSR_RBPN_V4_REF, FeatureExtractor
from data import get_test_set
from functools import reduce
import numpy as np
import utils
from math import ceil
import time
import cv2
import math
import pdb
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/test/Public/NTIRE2021/data/vsr')
parser.add_argument('--file_list', type=str, default='testlist.txt')
parser.add_argument('--model_type', type=str, default='FBPNSR_RBPN_V4_REF')
parser.add_argument('--output', default='./Results/test/test/ft4_epoch24/', help='Location to save checkpoint models')
parser.add_argument('--model', default='weights/4x_test-R8428-G11FBPNSR_RBPN_V4_REFVGG_FR_VIMEO_ft4_epoch_24.pth', help='sr pretrained base model')
parser.add_argument('--type', type=str, default='test')
parser.add_argument('--gt_path', type=str, default='/home/test/Public/NTIRE2021/data/vsr/val/val_sharp')


opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if opt.type=='test':
    opt.data_dir=os.path.join(opt.data_dir,'test','test_sharp_bicubic','X4')
elif opt.type=='val':
    opt.data_dir=os.path.join(opt.data_dir,'val','val_sharp_bicubic','X4')

print('===> Loading datasets')
test_set = get_test_set(opt.data_dir, opt.upscale_factor, opt.file_list)
print(test_set.__len__())
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
print(testing_data_loader.__len__())

print('===> Building model ', opt.model_type)
if opt.model_type == 'FBPNSR_RBPN_V1_REF':
    model = FBPNSR_RBPN_V1_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V2_REF':
    model = FBPNSR_RBPN_V2_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V3_REF':
    model = FBPNSR_RBPN_V3_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V4_REF':
    model = FBPNSR_RBPN_V4_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V1':
    model = FBPNSR_RBPN_V1(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V2':
    model = FBPNSR_RBPN_V2(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V3':
    model = FBPNSR_RBPN_V3(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V4':
    model = FBPNSR_RBPN_V4(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
    
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

print('---------- Networks architecture -------------')
#print_network(model)
print('----------------------------------------------')

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')


if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    print('star')
    model.eval()
    avg_psnr_predicted = 0.0
    index=1
    for batch in testing_data_loader:
        t0 = time.time()
        input, flow_f, flow_b , d_dir , filename = batch[0], batch[1], batch[2], batch[3], batch[4]
        print(d_dir)
        with torch.no_grad():
            t_im1 = Variable(input[0]).cuda(gpus_list[0])
            t_im2 = Variable(input[1]).cuda(gpus_list[0])
            t_flow_f = Variable(flow_f).cuda(gpus_list[0]).float()
            t_flow_b = Variable(flow_b).cuda(gpus_list[0]).float()
                        
        if opt.chop_forward:
            with torch.no_grad():
                pred_ht, pred_h1, pred_h2, pred_l  = chop_forward(t_im1, t_im2, t_flow_f, t_flow_b, model)
        else:
            with torch.no_grad():
                pred_ht, pred_h1, pred_h2, pred_l  = model(t_im1, t_im2, t_flow_f, t_flow_b, train=False)
            
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (d_dir[0]+'/frame10i11.png', (t1 - t0)))
        pred_ht = utils.denorm(pred_ht[0].cpu().data,vgg=True)
        pred_h1 = utils.denorm(pred_h1[0].cpu().data,vgg=True)
        pred_h2 = utils.denorm(pred_h2[0].cpu().data,vgg=True)
        #pred_l = utils.denorm(pred_l[0].cpu().data,vgg=True)
        
        index=index%98
        name=[]
        for i in range(index-1,index+2):
            a=str(i).zfill(8) +'.png'
            name.append(a)
        #save_img(pred, d_dir[0],'frame10i11.png', True)
        save_img(pred_ht, d_dir[0],name[1], False)
        save_img(pred_h1, d_dir[0],name[0], False)
        save_img(pred_h2, d_dir[0],name[2], False)
        #save_img(pred_l, d_dir[0],'im_l.png', False)
        #save_img(target, str(count), False)
        
        # if opt.type=='val':
        #     folder_list=os.listdir(opt.gt_path)
        #     folder_list.sort()
        #     target_path=os.path.join(opt.gt_path,folder_list[i])
        #     target_list=os.listdir(target_path)
        #     target_list.sort()
        #     SR_img = ht
        #     target_img = imageio.imread(os.path.join(target_path,target_list[index]))
        #     psnr = peak_signal_noise_ratio(SR_img, target_img) #psnr
        #     ssim = structural_similarity(SR_img, target_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        #     print('PSNR:',psnr)
        #     print('SSIM:',ssim)
        #     SR_img = h1
        #     target_img = imageio.imread(os.path.join(target_path,target_list[index-1]))
        #     psnr_ = peak_signal_noise_ratio(SR_img, target_img) #psnr
        #     ssim = structural_similarity(SR_img, target_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        #     print('PSNR:',psnr)
        #     print('SSIM:',ssim)
            
            
        
    #print("PSNR_predicted=", avg_psnr_predicted/count)
        index=index+2
def save_img(img, d_dir,img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    print(d_dir)
    filename = os.path.splitext(img_name)
    # save img
    save_dir=os.path.join(opt.output, d_dir)
    print(opt.output,'-----',save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if pred_flag:
        save_fn = save_dir +'/'+ filename[0]+'_'+opt.model_type+filename[1]
        print('t')
    else:
        print('f')
        save_fn = opt.output+d_dir+ '/'+img_name
        
    print(save_fn)
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
       
def chop_forward(t_im1, t_im2, t_flow_f, t_flow_b, iter, model, shave=8, min_size=160000, nGPUs=opt.gpus):
    b, c, h, w = t_im1.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    
    mod_size = 4
    if h_size%mod_size:
        h_size = ceil(h_size/mod_size)*mod_size
    if w_size%mod_size:
        w_size = ceil(w_size/mod_size)*mod_size
        
    inputlist = [
        [t_im1[:, :, 0:h_size, 0:w_size], t_im2[:, :, 0:h_size, 0:w_size], t_flow_f[:, :, 0:h_size, 0:w_size], t_flow_b[:, :, 0:h_size, 0:w_size], iter],
        [t_im1[:, :, 0:h_size, (w - w_size):w],t_im2[:, :, 0:h_size, (w - w_size):w],t_flow_f[:, :, 0:h_size, (w - w_size):w],t_flow_b[:, :, 0:h_size, (w - w_size):w],iter ],
        [t_im1[:, :, (h - h_size):h, 0:w_size],t_im2[:, :, (h - h_size):h, 0:w_size],t_flow_f[:, :, (h - h_size):h, 0:w_size],t_flow_b[:, :, (h - h_size):h, 0:w_size],iter ],
        [t_im1[:, :, (h - h_size):h,  (w - w_size):w],t_im2[:, :, (h - h_size):h,  (w - w_size):w],t_flow_f[:, :, (h - h_size):h,  (w - w_size):w],t_flow_b[:, :, (h - h_size):h,  (w - w_size):w],iter ]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch[0], input_batch[1], input_batch[2], input_batch[3], train=False)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2],patch[3],patch[4], model, shave, min_size, nGPUs) \
            for patch in inputlist]

    scale=1
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(t_im1.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output



##Eval Start!!!!
eval()
