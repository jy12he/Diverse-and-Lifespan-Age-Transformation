### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import scipy # this is to prevent a potential error caused by importing torch before scipy (happens due to a bad combination of torch & scipy versions)
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models_distan.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from pdb import set_trace as st

import wrap
import cv2

def test(opt):
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test batches = %d' % (int(dataset_size / len(opt.sort_order))))
    visualizer = Visualizer(opt)

    model = create_model(opt)
    model.eval()

    # create webpage  --random_seed', type=int, default=-1
    if opt.random_seed != -1:
        exp_dir = '%s_%s_seed%s' % (opt.phase, opt.which_epoch, str(opt.random_seed))
    else:
        exp_dir = '%s_%s' % (opt.phase, opt.which_epoch)
    web_dir = os.path.join(opt.results_dir, opt.name, exp_dir)
    # print(web_dir)  ./results_test/males_model/test_20

# --traverse', action='store_true'    --deploy', action='store_true'
    if opt.traverse or opt.deploy:
        if opt.traverse:
            out_dirname = 'traversal'
        else:
            out_dirname = 'deploy'
        output_dir = os.path.join(web_dir,out_dirname)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for image_path in opt.image_path_list:
            #import ipdb; ipdb.set_trace()
            #print(image_path)
            data = dataset.dataset.get_item_from_path(image_path)

            visuals = model.inference(data)

            if opt.traverse and opt.make_video:
                out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.mp4')
                visualizer.make_video(visuals, out_path)
            elif opt.traverse or (opt.deploy and opt.full_progression):
                if opt.traverse and opt.compare_to_trained_outputs:
                    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_compare_to_{}_jump_{}.png'.format(opt.compare_to_trained_class, opt.trained_class_jump))
                else:
                    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
                visualizer.save_row_image(visuals, out_path, traverse=opt.traverse)
            else:
                out_path = os.path.join(output_dir, os.path.basename(image_path[:-4]))
                visualizer.save_images_deploy(visuals, out_path)
    else:
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        # print(opt.name, opt.phase, opt.which_epoch)  males_model test 20

# /////////////////////////////////////////////////////////
        # test
        processed_num = 0
        for i, data in enumerate(dataset):

            #import ipdb; ipdb.set_trace()
            if i >= opt.how_many:
                break
#  ////////////////////////////////////////////////////
            # visuals = model.inference(data, out_dir=opt.results_dir) #   1111

 
   
            select_two_age = 5 
    

            visuals = model.inference(data, web_dir, select_two_age)  
            
            img_path = data['Paths']

            
            img_path = img_path[select_two_age:select_two_age + 1]
          
            rem_ind = []
            wrap_image_name = []
            for i, path in enumerate(img_path):
                if path != '':
                    processed_num = processed_num + 1
                    print('process image... %s   %d' % (path, processed_num))
                    if i < 2:
                        image_name = (path.split('/')[-1]).split('.')[0]
                        wrap_image_name.append(image_name)
                else:
                    rem_ind += [i]

            for ind in reversed(rem_ind):
                del img_path[ind]
              
            visualizer.save_images(webpage, visuals, img_path)
            webpage.save()
            
    
     # wrap 
   
            # img_dir = os.path.join(web_dir, 'images')
            # landmarks_dir = os.path.join(web_dir, 'landmarks')
            # wrap_result_dir = os.path.join(web_dir, 'wrap_result')
            # if not os.path.exists(wrap_result_dir):
            #     os.makedirs(wrap_result_dir)
            
            # for i in range(6):
            #     wrap_age_dir = os.path.join(wrap_result_dir, str(i))
            #     if not os.path.exists(wrap_age_dir):
            #         os.makedirs(wrap_age_dir)
            #         for m in range(3):   # 存放同一张图片的 多样性结果
            #             wrap_div_dir = os.path.join(wrap_age_dir, str(m))
            #             if not os.path.exists(wrap_div_dir):
            #                 os.makedirs(wrap_div_dir)

            # for img_name in wrap_image_name:
            #     for age in range(6):
            #         for i in range(1):  # 每个结果生成3张图像  测试指标是1张图像(average latent code)
            #             img_path = img_dir + '/' + img_name +'_tex_trans_to_class_'+str(age)+'_' + str(2)+'.png'   # str(i)
            #             for m in range(3):  # 每个结果生成 4 个 landmarks  测试指标是1个(average landmarks)
            #                 land_age = age
            #                 # 50-69的landmarks ， 用和 30-39 一样的
            #                 if age==5:
            #                     land_age = 4
            #                 landmarks_path = landmarks_dir + '/' + img_name + '_trans_to_class_' + str(land_age) + '_' + str(m)+'.txt'
            #                 # print(landmarks_path)                    str(m)

            #                 wrap_result = wrap.img_txt_wrap(img_path, landmarks_path)
                          
            #                 # wrap_img_path = wrap_result_dir + '/' + img_name + '_trans_to_class_' + str(age) + '_from_' + str(i) + '_and_' + str(m) + '.png'
                           
            #                 wrap_img_path = wrap_result_dir + '/' + str(age) + '/' + str(m) + '/' + img_name + '.png'
            #                 # print(wrap_img_path)
            #                 cv2.imwrite(wrap_img_path, wrap_result)
            #                 # print(str(i), str(m))
            #                 print('wrapping image %d  and  landmarks %d' % (i, m))
    



if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    test(opt)
