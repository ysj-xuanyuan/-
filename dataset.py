import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from skimage import color
from random import randrange
import os.path
import cv2

max_flow = 150.0 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img_ntire2021(filepath, scale):
    # Get a triplet
    ##list=os.listdir(filepath)
    filedir, filename = os.path.split(filepath)#分离文件名和文件夹名字
    id_current, _ = os.path.splitext(filename)#分离文件名字和扩展名
    idx = int(id_current)
    if idx == 0:
        # The current frame is the begin frame of the video
        _list = [join(filedir, '%08d.png'%k) for k in range(idx, idx+3)] # %代表格式化输出，08d分别代表占位8位和整数
    elif idx == 99:
        # The current frame is the end frame of the video
        _list = [join(filedir, '%08d.png'%k) for k in range(idx-3+1, idx+1)]# from os.path import join
    else:
        # The current frame is the middle frame of the video
        _list = [join(filedir, '%08d.png'%k) for k in range(idx-1, idx+2)]

    #print('\t\t--list: ', len(_list), _list)
    _list.sort()
    #print('\t\t--list: ', len(_list), _list)
    
    rate = 1
    #for vimeo90k-setuplet (multiple temporal scale)
    #if random.random() < 0.5:
    #    rate = 2
    
    #print('\t\t--0, %d'%(len(_list)-(2*rate)))
    index = random.randrange(0, len(_list)-(2*rate))  #生成
    #index = 0
    #print('\t\t--randrange: ', 0, len(_list) - (2* rate))
    #print('\t\t--index: ', index)

    #print('\t\t--target: ', index, index + 3* rate, rate)
    for k in range(index, index+3*rate, rate):
        #print('\t\t--', k)
        imgfile = _list[k].replace('_bicubic', '').replace('X4/', '')
        #print('\t\t--', imgfile)
        img = Image.open(imgfile).convert('RGB')
        #print('\t\t--', np.shape(img))

    target = [modcrop(Image.open(_list[i].replace('_bicubic', '').replace('X4/', '')).convert('RGB'), scale) for i in range(index, index+3*rate, rate)]
    #replace   str.replace(old, new)
    #print('\t\t--target: ', target)
    
    h,w = target[0].size
    h_in,w_in = int(h//scale), int(w//scale)
    #print('\t\t--target: ', w, h)
    #print('\t\t--in: ', w_in, h_in)
    
    # Get the low_resolution output target
    #target_l = target[1].resize((h_in,w_in), Image.BICUBIC)
    target_l = modcrop(Image.open(_list[1]).convert('RGB'), scale)
    #print('\t\t--target_l: ', np.shape(target_l), target_l)

    # Get the input 2-frame data
    #input = [target[j].resize((h_in,w_in), Image.BICUBIC) for j in [0,2]]
    _input  = [modcrop(Image.open(_list[i]).convert('RGB'), scale) for i in [0, 2]]
    #print('\t\t--input: ', _input)
    
    return _input, target, target_l, _list

def load_img(filepath, scale):
    list=os.listdir(filepath)
    #print('\t\t--list: ', len(list), list)
    list.sort()
    #print('\t\t--list: ', list)
    
    rate = 1
    #for vimeo90k-setuplet (multiple temporal scale)
    #if random.random() < 0.5:
    #    rate = 2
    
    index = randrange(0, len(list)-(2*rate))
    #print('\t\t--randrange: ', 0, len(list) - (2* rate))
    
    #print('\t\t--target: ', index, index + 3* rate)
    target = [modcrop(Image.open(filepath+'/'+list[i]).convert('RGB'), scale) for i in range(index, index+3*rate, rate)]
    #print('\t\t--target: ', target)
    
    h,w = target[0].size
    h_in,w_in = int(h//scale), int(w//scale)
    #print('\t\t--target: ', w, h)
    #print('\t\t--in: ', w_in, h_in)
    
    target_l = target[1].resize((h_in,w_in), Image.BICUBIC)
    #print('\t\t--target_l: ', np.shape(target_l), target_l)

    input = [target[j].resize((h_in,w_in), Image.BICUBIC) for j in [0,2]]
    #print('\t\t--input: ', input)
    
    return input, target, target_l, list

def load_img_test_ntire2021(filepath, scale):
    rate = 1
    # Get a triplet
    ##list=os.listdir(filepath)
    filedir, filename = os.path.split(filepath)
    id_current, _ = os.path.splitext(filename)
    idx = int(id_current)
    # The current frame is the begin frame of the video
    list = [join(filedir, '%08d.png'%k) for k in range(idx, idx+2)]

    #print('\t\t--list: ', len(list), list)
    list.sort()
    #print('\t\t--list: ', len(list), list)
    
    input = [modcrop(Image.open(list[i]).convert('RGB'), scale) for i in range(len(list))]
    
    return input, list

def load_img_test(filepath, scale):
    list=os.listdir(filepath)
    list.sort()
    
    target = [modcrop(Image.open(filepath+'/'+list[i]).convert('RGB'), scale) for i in range(len(list))]
    h,w = target[0].size
    h_in,w_in = int(h//scale), int(w//scale)
    
    input = [target[j].resize((h_in,w_in), Image.BICUBIC) for j in [0,len(list)-1]]
    
    return input, list

def load_img_nodown(filepath):
    list=os.listdir(filepath)
    list.sort()
    
    input = [Image.open(filepath+'/'+list[i]).convert('RGB') for i in [0,len(list)-1]]
    
    return input, list
    
def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75 #0.95 #0.75
    minWidth = 20 #50 #20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    
    #Rescale
    flow = rescale_flow(flow,-1,1)
    return flow

def rescale_flow(x,max_range,min_range):
    #remove noise
    x[x > max_flow] = max_flow
    x[x < -max_flow] = -max_flow
    
    max_val = max_flow 
    min_val = -max_flow 
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_tar_l, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_in] 
    img_tar = [j.crop((ty,tx,ty + tp, tx + tp)) for j in img_tar] 
    img_tar_l = img_tar_l.crop((iy,ix,iy + ip, ix + ip)) 
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_tar_l, info_patch

def augment(img_in, img_tar, img_tar_l, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = [ImageOps.flip(j) for j in img_in]
        img_tar = [ImageOps.flip(j) for j in img_tar]
        img_tar_l = ImageOps.flip(img_tar_l)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = [ImageOps.mirror(j) for j in img_in]
            img_tar = [ImageOps.mirror(j) for j in img_tar]
            img_tar_l = ImageOps.mirror(img_tar_l)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = [j.rotate(180) for j in img_in]
            img_tar = [j.rotate(180) for j in img_tar]
            img_tar_l = img_tar_l.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_tar_l, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
                    
        if self.transform:
            input = [self.transform(j) for j in input]
            target = [self.transform(j) for j in target]
            target_l = self.transform(target_l)
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 

        return input, target, target_l, flow_f, flow_b, file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderFlowNTIRE2021(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolderFlowNTIRE2021, self).__init__()
        #print('\timage_dir:', image_dir)
        #print('\tupscale_factor:', upscale_factor)
        #print('\tdata_augmentation:', data_augmentation)
        #print('\tfile_list:', file_list)
        #print('\tpatch_size:', patch_size)

        ## Get the file list of all triplets
        #alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        lr_dir = os.path.join(image_dir, 'train_sharp_bicubic/X4')
        vlist = os.listdir(lr_dir)
        lr_filenames = []
        for vdir in vlist:
            ##print('\t\tvdir: ', vdir)
            alist = os.listdir(join(lr_dir, vdir))
            for img_name in alist:
                ##print('\t\t\t--img_name: ', img_name)
                lr_filenames.append(join(lr_dir, vdir, img_name))

        self.image_filenames = lr_filenames
        #print('\tself.image_filenames:', self.image_filenames)

        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        #print('\t==>load %04d-th image: %s'%(index, self.image_filenames[index]))
        input, target, target_l, file_list = load_img_ntire2021(self.image_filenames[index], self.upscale_factor)
        #print('\t\tinput: {}, {}, {}'.format(type(input), len(input), input))
        #print('\t\ttarget: {}, {}, {}'.format(type(target), len(target), target))
        #print('\t\ttarget_l: {}, {}, {}'.format(type(target_l), len([1]), target_l))

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        gt_flow_f = get_flow(input[0],target_l) + get_flow(target_l,input[1])
        gt_flow_b = get_flow(input[1],target_l) + get_flow(target_l,input[0])
                    
        if self.transform:
            input = [self.transform(j) for j in input]
            target = [self.transform(j) for j in target]
            target_l = self.transform(target_l)
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            gt_flow_f = torch.from_numpy(gt_flow_f.transpose(2,0,1)) 
            gt_flow_b = torch.from_numpy(gt_flow_b.transpose(2,0,1)) 

        return input, target, target_l, flow_f, flow_b, gt_flow_f, gt_flow_b,file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
        
class DatasetFromFolderFlow(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolderFlow, self).__init__()
        #print('\timage_dir:', image_dir)
        #print('\tupscale_factor:', upscale_factor)
        #print('\tdata_augmentation:', data_augmentation)
        #print('\tfile_list:', file_list)
        #print('\tpatch_size:', patch_size)

        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        #print('\talist:', alist)

        self.image_filenames = [join(image_dir,x) for x in alist]
        #print('\tself.image_filenames:', self.image_filenames)

        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        #print('\t==>load %04d-th image: %s'%(index, self.image_filenames[index]))
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)
        #print('\t\tinput: {}, {}, {}'.format(type(input), len(input), input))
        #print('\t\ttarget: {}, {}, {}'.format(type(target), len(target), target))
        #print('\t\ttarget_l: {}, {}, {}'.format(type(target_l), len([1]), target_l))

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        gt_flow_f = get_flow(input[0],target_l) + get_flow(target_l,input[1])
        gt_flow_b = get_flow(input[1],target_l) + get_flow(target_l,input[0])
                    
        if self.transform:
            input = [self.transform(j) for j in input]
            target = [self.transform(j) for j in target]
            target_l = self.transform(target_l)
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            gt_flow_f = torch.from_numpy(gt_flow_f.transpose(2,0,1)) 
            gt_flow_b = torch.from_numpy(gt_flow_b.transpose(2,0,1)) 

        return input, target, target_l, flow_f, flow_b, gt_flow_f, gt_flow_b,file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
        
class DatasetFromFolderFlowLR(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolderFlowLR, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(target[0],target[2])
        flow_b = get_flow(target[2],target[0])
        
        gt_flow_f = get_flow(target[0],target[1]) + get_flow(target[1],target[2])
        gt_flow_b = get_flow(target[2],target[1]) + get_flow(target[1],target[0])
                    
        if self.transform:
            target = [self.transform(j) for j in target]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            gt_flow_f = torch.from_numpy(gt_flow_f.transpose(2,0,1)) 
            gt_flow_b = torch.from_numpy(gt_flow_b.transpose(2,0,1)) 
            

        return target, flow_f, flow_b, gt_flow_f, gt_flow_b, file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
    
class DatasetFromFolderTestNTIRE2021(data.Dataset):
    def __init__(self, image_dir, upscale_factor, file_list, transform=None, track=2):
        super(DatasetFromFolderTestNTIRE2021, self).__init__()
        ## Get the file list of all triplets
        #alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        #self.image_filenames = [join(image_dir,x) for x in alist]
        lr_dir = os.path.join(image_dir, 'test_sharp_bicubic')
        vlist = os.listdir(lr_dir)
        lr_filenames = []
        for vdir in vlist:
            alist = os.listdir(vdir)
            for img_name in alist:
                lr_filenames.append(join(lr_dir, vdir, img_name))

        if track == 2:
            self.image_filenames = lr_filenames[::2]
        else:
            self.image_filenames = lr_filenames
        #print('==>image_filenames:\n', self.image_filenames)
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input, file_list = load_img_test_ntire2021(self.image_filenames[index], self.upscale_factor, track)
        #print('\t\t--index: {}'.format(index))
        #print('\t\t--file_list:\n', file_list)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        if self.transform:
            input = [self.transform(j) for j in input]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            
        return input, flow_f, flow_b, file_list, self.image_filenames[index]
      
    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, upscale_factor, file_list, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        #print('==>alist:\n', alist)
        self.image_filenames = [join(image_dir,x) for x in alist]
        #print('==>image_filenames:\n', self.image_filenames)
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input, file_list = load_img_test(self.image_filenames[index], self.upscale_factor)
        #print('\t\t--index: {}'.format(index))
        #print('\t\t--file_list:\n', file_list)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        if self.transform:
            input = [self.transform(j) for j in input]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            
        return input, flow_f, flow_b, file_list, self.image_filenames[index]
      
    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderInterp(data.Dataset):
    def __init__(self, image_dir, file_list, transform=None):
        super(DatasetFromFolderInterp, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.transform = transform

    def __getitem__(self, index):
        input, file_list = load_img_nodown(self.image_filenames[index])
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        if self.transform:
            input = [self.transform(j) for j in input]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            
        return input, flow_f, flow_b, file_list, self.image_filenames[index]
      
    def __len__(self):
        return len(self.image_filenames)
    
