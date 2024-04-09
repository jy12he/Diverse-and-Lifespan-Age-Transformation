### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from pdb import set_trace as st

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_txt_file(filename):
    return filename.endswith('.txt')


def list_folder_images(dir, opt):
    images = []
    parsings = []
    landmarks = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
#  dir: ./datasets/males/train0-2
    for fname in os.listdir(dir+'/landmarks_81'):    # ./datasets/males/train0-2/landmarks_81下的文件名
        if is_txt_file(fname):
            landmarks_path = os.path.join(dir+'/landmarks_81', fname)  # ./datasets/males/train0-2/landmarks_81/0000.txt
    
            path = os.path.join(dir, fname[:-3] + 'png')  # ./datasets/males/train0-2/0000.png
            parsing_fname = fname[:-3] + 'png'

            if os.path.isfile(os.path.join(dir, 'parsings', parsing_fname)):
                parsing_path = os.path.join(dir, 'parsings', parsing_fname)
                images.append(path)
                parsings.append(parsing_path)
                landmarks.append(landmarks_path)
             

    # sort according to identity in case of FGNET test
    if 'fgnet' in opt.dataroot.lower():
        images.sort(key=str.lower)
        parsings.sort(key=str.lower)

    return images, parsings, landmarks

def get_transform(opt, normalize=True):
    transform_list = []
    print('resize_or_crop: {}'.format(opt.resize_or_crop))
    print('no_flip: {}'.format(opt.no_flip))

    if opt.resize_or_crop == 'resize_and_crop':   # resize_or_crop: resize_and_crop
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, interpolation=Image.NEAREST))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))

# 1
    # if opt.isTrain and not opt.no_flip:  # flip 
    #     transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]

    if normalize:
        mean = (0.5,)
        std = (0.5,)
        transform_list += [transforms.Normalize(mean,std)]

    return transforms.Compose(transform_list)
