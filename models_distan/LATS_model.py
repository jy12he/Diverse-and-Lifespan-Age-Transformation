### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import cv2
import numpy as np
import dlib
import random
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import torchvision

import re
import functools
from collections import OrderedDict
from .base_model import BaseModel
import util.util as util
from . import networks
from pdb import set_trace as st

# from deepface import DeepFace  
import time

import torchvision.models as models
import os
# arc face
from arcface.models import *
from torch.nn import DataParallel

from torchinfo import summary 

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
         
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)

        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            "relu1_1": relu1_1,
            "relu1_2": relu1_2,
            "relu2_1": relu2_1,
            "relu2_2": relu2_2,
            "relu3_1": relu3_1,
            "relu3_2": relu3_2,
            "relu3_3": relu3_3,
            "max_3": max_3,
            "relu4_1": relu4_1,
            "relu4_2": relu4_2,
            "relu4_3": relu4_3,
            "relu5_1": relu5_1,
            "relu5_2": relu5_2,
            "relu5_3": relu5_3,
        }
        return out

#  1
class PerceptualLoss(nn.Module):

    def __init__(self, weights=[0.05, 0.05, 0.05, 0.05, 0.05]):
    # def __init__(self, weights=[0.0, 0.0, 0.1, 0.1, 0.1]):
        super(PerceptualLoss, self).__init__()
        self.add_module("vgg", VGG16().cuda())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(
            x_vgg["relu1_1"], y_vgg["relu1_1"]
        )
        content_loss += self.weights[1] * self.criterion(
            x_vgg["relu2_1"], y_vgg["relu2_1"]
        )
        content_loss += self.weights[2] * self.criterion(
            x_vgg["relu3_1"], y_vgg["relu3_1"]
        )
        content_loss += self.weights[3] * self.criterion(
            x_vgg["relu4_1"], y_vgg["relu4_1"]
        )
        content_loss += self.weights[4] * self.criterion(
            x_vgg["relu5_1"], y_vgg["relu5_1"]
        )
        return content_loss


class Diversityloss(nn.Module):
    def __init__(self):
        super(Diversityloss, self).__init__()
        self.vgg = VGG16().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        diversity_loss = 0.0
        diversity_loss += self.weights[4] * self.criterion(
            x_vgg["relu4_1"], y_vgg["relu4_1"]
        )
        return diversity_loss  
#####################

class LATS(BaseModel): #Lifetime Age Transformation Synthesis
    def name(self):
        return 'LATS'


    def initialize(self, opt):
        #import ipdb; ipdb.set_trace()
        BaseModel.initialize(self, opt)

        # 1
        self.mean_face()  

        self.face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

      
        self.Diversityloss = Diversityloss()
     
        self.PerceptualLoss = PerceptualLoss()


        self.boundary = np.array([[0, 0, 0, 0, 0, 64, 128, 192, 255, 64, 128, 192, 255, 255, 255, 255],
                         [0, 64, 128, 192, 255, 255, 255, 255, 255, 0, 0, 0, 0, 64, 128, 192]], dtype=float)
       


        self.PCA = torch.tensor(np.loadtxt('/data/DLFS/DLFS-main/V81.txt'), dtype=torch.float64).cuda()
        self.mean = torch.tensor(np.loadtxt('/data/DLFS/DLFS-main/means81.txt'), dtype=torch.float64).cuda()
        
        self.PCA_32 = torch.tensor(np.loadtxt('/data/DLFS/DLFS-main/V81.txt'), dtype=torch.float32).cuda()
        self.mean_32 = torch.tensor(np.loadtxt('/data/DLFS/DLFS-main/means81.txt'), dtype=torch.float32).cuda()
        
        self.detector = dlib.get_frontal_face_detector()
        self.pre = dlib.shape_predictor("/data/DLFS/DLFS-main/shape_predictor_68_face_landmarks.dat")
       
    
      
        torch.backends.cudnn.benchmark = True

        # determine mode of operation [train, test, deploy, traverse (latent interpolation)]
        self.isTrain = opt.isTrain
        self.traverse = (not self.isTrain) and opt.traverse

        # mode to generate Fig. 15 in the paper
        self.compare_to_trained_outputs = (not self.isTrain) and opt.compare_to_trained_outputs
        if self.compare_to_trained_outputs:
            self.compare_to_trained_class = opt.compare_to_trained_class
            self.trained_class_jump = opt.trained_class_jump

        self.deploy = (not self.isTrain) and opt.deploy
        if not self.isTrain and opt.random_seed != -1:
            torch.manual_seed(opt.random_seed)
            torch.cuda.manual_seed_all(opt.random_seed)
            np.random.seed(opt.random_seed)

        # network architecture parameters
        self.nb = opt.batchSize
        self.size = opt.fineSize
        self.ngf = opt.ngf
        self.ngf_global = self.ngf

        self.numClasses = opt.numClasses
        self.use_moving_avg = not opt.no_moving_avg

        self.no_cond_noise = opt.no_cond_noise
        style_dim = opt.gen_dim_per_style * self.numClasses # 50 * 6
        self.duplicate = opt.gen_dim_per_style

        self.cond_length = style_dim

        # self.active_classes_mapping = opt.active_classes_mapping

        if not self.isTrain:
            self.debug_mode = opt.debug_mode
        else:
            self.debug_mode = False
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_fair_4 = torchvision.models.resnet34(pretrained=True)
        model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
        model_fair_4.load_state_dict(torch.load('./res34_fair_align_multi_4_20190809.pt'))
        self.model_fair_4 = model_fair_4.to(device)


        self.arc_model = resnet_face18(False)
        self.arc_model = DataParallel(self.arc_model)
        self.arc_model.load_state_dict(torch.load('./arcface/resnet18_110.pth'))
        self.arc_model.to(torch.device("cuda"))

        ##### define networks
        # Generators network
        if opt.encoder_type == 'original':

            self.netG = self.parallelize(networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                     id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids, padding_type='reflect', style_dim=style_dim,
                                     init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                     decoder_norm=opt.decoder_norm, activation=opt.activation,
                                     adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                     modulated_conv=opt.use_modulated_conv))
        elif opt.encoder_type == 'distan':

            self.netG = self.parallelize(networks.define_distan_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                     id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids, padding_type='reflect', style_dim=style_dim,
                                     init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                     decoder_norm=opt.decoder_norm, activation=opt.activation,
                                     adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                     modulated_conv=opt.use_modulated_conv))
       
            self.netLandG = networks.define_landmarks_G()
    
            self.style_encoder_L = networks.define_StyleEncoder_L() 
  
            self.mapping_net_L = networks.define_MappingNet_L()


         
            self.style_encoder = networks.define_StyleEncoder()  # 1
          
            self.mapping_net = networks.define_MappingNet()

            # self.LandDis = networks.define_LandD()

        if self.isTrain and self.use_moving_avg:
            if opt.encoder_type == 'original':

                self.g_running = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                                   id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids, padding_type='reflect', style_dim=style_dim,
                                                   init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                                   decoder_norm=opt.decoder_norm, activation=opt.activation,
                                                   adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                                   modulated_conv=opt.use_modulated_conv)
            elif opt.encoder_type == 'distan':
                self.g_running = networks.define_distan_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                                   id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids, padding_type='reflect', style_dim=style_dim,
                                                   init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                                   decoder_norm=opt.decoder_norm, activation=opt.activation,
                                                   adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                                   modulated_conv=opt.use_modulated_conv)
            self.g_running.train(False)
            self.requires_grad(self.g_running, flag=False)  
           
            self.accumulate(self.g_running, self.netG, decay=0) 

        # Discriminator network
        if self.isTrain:
            self.netD = self.parallelize(networks.define_D(opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                         numClasses=self.numClasses, gpu_ids=self.gpu_ids,
                                         init_type='kaiming'))
            #if opt.encoder_type == 'distan':
            #    self.cooccur_D = self.parallelize(networks.define_cooccur_D(32, gpu_ids=self.gpu_ids))

        if self.opt.verbose:
                print('---------- Networks initialized -------------')


        # load networks
        if (not self.isTrain) or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if (not self.isTrain) or (self.isTrain and opt.continue_train) else opt.load_pretrain
            if self.isTrain:
                # self.load_network(self.netG, 'G_tex', opt.which_epoch, pretrained_path)
                # self.load_network(self.netD, 'D_tex', opt.which_epoch, pretrained_path)
                self.load_network(self.mapping_net, 'Mapping', opt.which_epoch, pretrained_path)
                self.load_network(self.style_encoder, 'Style_encoder', opt.which_epoch, pretrained_path)
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)

                if self.use_moving_avg:
                    # self.load_network(self.g_running, 'g_running', opt.which_epoch, pretrained_path)
                    self.load_network(self.g_running, 'g_running', opt.which_epoch, pretrained_path)
 

            elif self.use_moving_avg:
                self.load_network(self.netG, 'g_running', opt.which_epoch, pretrained_path)
                # 1
                self.load_network(self.mapping_net, 'Mapping', opt.which_epoch, pretrained_path)
                self.load_network(self.style_encoder, 'Style_encoder', opt.which_epoch, pretrained_path)
    
         
                landmarks_save_dir = os.path.join(opt.land_checkpoints_dir, opt.name)

                self.load_network(self.netLandG, 'netLandG', 20, landmarks_save_dir)
                self.load_network(self.mapping_net_L, 'mapping_net_L', 20, landmarks_save_dir)

            else:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                 # 1
                self.load_network(self.mapping_net, 'Mapping', opt.which_epoch, pretrained_path)
                self.load_network(self.style_encoder, 'Style_encoder', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            # define loss functions
            self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
            self.R1_reg = networks.R1_reg()
            self.age_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.identity_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.struct_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.texture_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.criterionCycle = self.parallelize(networks.FeatureConsistency()) #torch.nn.L1Loss()
            self.criterionRec = self.parallelize(networks.FeatureConsistency()) #torch.nn.L1Loss()
            # 1
            self.criteriondiv = self.parallelize(networks.FeatureConsistency())
            self.criterionStyleRe = self.parallelize(networks.FeatureConsistency())
            self.criterionshape = self.parallelize(networks.FeatureConsistency())

            # initialize optimizers
            self.old_lr = opt.lr

            # 1  set optimizer  Land
            paramsL = list(self.netLandG.parameters())
            self.optimizer_L = torch.optim.Adam(paramsL, lr=0.00001, betas=(opt.beta1, opt.beta2))

            paramsStyE = list(self.style_encoder.parameters())
            self.optimizer_StyE = torch.optim.Adam(paramsStyE, lr=0.0001, betas=(0.0, 0.99),weight_decay=1e-4)

            paramsM = list(self.mapping_net.parameters())
            self.optimizer_M = torch.optim.Adam(paramsM, lr=1e-6, betas=(0.0, 0.99),weight_decay=1e-4)

            # set optimizer G
            paramsG = []
            params_dict_G = dict(self.netG.named_parameters())
            # set the MLP learning rate to 0.01 or the global learning rate
            for key, value in params_dict_G.items():
                decay_cond = ('decoder.mlp' in key)
                if opt.decay_adain_affine_layers:
                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                if decay_cond:
                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                else:
                    paramsG += [{'params':[value],'lr':opt.lr}]

            self.optimizer_G = torch.optim.Adam(paramsG, lr=opt.lr, betas=(opt.beta1, opt.beta2))

            # set optimizer D
            if opt.encoder_type == 'original':

                paramsD = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(paramsD, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            if opt.encoder_type == 'distan':
                paramsD = list(self.netD.parameters()) #+ list(self.cooccur_D.parameters())
                self.optimizer_D = torch.optim.Adam(paramsD, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def WarpMapping(self, srt_img, srt_landmark, dst_landmark):
        num_ctrl_points = 84
        pad_landmarks = np.zeros((84-68, 2), np.uint8)
        srt_landmark = np.vstack((srt_landmark, pad_landmarks))
        dst_landmark = np.vstack((dst_landmark, pad_landmarks))
      
        for i in range(0, 16):
            srt_landmark[i + 64, 0] = self.boundary[0, i]
            srt_landmark[i + 64, 1] = self.boundary[1, i]
            dst_landmark[i + 64, 0] = self.boundary[0, i]
            dst_landmark[i + 64, 1] = self.boundary[1, i]
        
        # CalculateCoeff
        num = srt_landmark.shape[0] 
        A = np.zeros((num + 3, num + 3))
        b = np.zeros((num + 3, 2))
        wv = np.ones((num_ctrl_points + 3, 2))
        for i in range(0, num):
            row = num - i
            tmp_mat1 = dst_landmark[i:i+row, 0] - dst_landmark[i, 0]
            tmp_mat2 = dst_landmark[i:i+row, 1] - dst_landmark[i, 1]
            A[i:i+row, i] = np.sqrt(np.square(tmp_mat1) + np.square(tmp_mat2))
            tmp_mat1 = A[i:i+row, i]
            A[i, i:i+row] = tmp_mat1.T
        A[num, 0:num] = np.ones((1, num), dtype=float)
        A[0:num, num:num+1] = np.ones((num, 1), dtype=float)
        A[num+1:num+3, 0:num] = dst_landmark.T
        A[0:num, num+1:num+3] = dst_landmark
        b[0:num, 0:2] = srt_landmark - dst_landmark
        wv[0:num+3, 0] = np.squeeze(np.linalg.lstsq(A, b[0:num+3, 0], rcond=-1)[0])
        wv[0:num+3, 1] = np.squeeze(np.linalg.lstsq(A, b[0:num+3, 1], rcond=-1)[0])

        # image wrap
        dst = np.zeros((256, 256, 3), np.uint8)
        h = dst.shape[0]
        w = dst.shape[1]
        num = dst_landmark.shape[0]
        
        new_x_mat = np.zeros((256, 256))
        new_y_mat = np.zeros((256, 256))
        base_mat = np.zeros((256, 256))
        tmp_mat = np.zeros((256, 256))
        tmp_vec = np.zeros(256)
        for i in range(256):
            tmp_mat[0:256, i:i+1] = i * np.ones((256, 1))
            tmp_vec[i] = i
        for i in range(num):
            base_l = np.square(tmp_mat - dst_landmark[i, 0])
            base_r = np.tile(np.square(tmp_vec - dst_landmark[i, 1]).reshape(256, 1), 256)
            base_mat = np.sqrt(base_l + base_r)
            new_x_mat = new_x_mat + base_mat * wv[i, 0]
            new_y_mat = new_y_mat + base_mat * wv[i, 1]

        new_x_mat = (new_x_mat + wv[num + 2, 0] * tmp_mat.T
                    + (wv[num + 1, 0] + 1) * tmp_mat) + wv[num,0]
        new_y_mat = (new_y_mat + (wv[num + 2, 1] + 1) * tmp_mat.T
                    + wv[num + 1, 1] * tmp_mat) + wv[num,1]
        
        for i in range(256):
            for j in range(256):
                x = new_x_mat[i][j]
                y = new_y_mat[i][j]
                floor_x = int(np.floor(x))
                floor_y = int(np.floor(y))
                if ((floor_x < 0) or ((floor_x + 1) >= w) or (floor_y < 0) or ((floor_y + 1) >= h)):
                    continue
                u = x - floor_x
                v = y - floor_y

                dst[i][j][0] = (1-u) * (1-v) * int(srt_img[floor_y][floor_x][0]) + u * (1 - v) * int(srt_img[floor_y][floor_x + 1][0]) \
                                + v * (1-u) * int(srt_img[floor_y + 1][floor_x][0]) + u * v * int(srt_img[floor_y + 1][floor_x + 1][0])

                dst[i][j][1] = (1-u) * (1-v) * int(srt_img[floor_y][floor_x][1]) + u * (1 - v) * int(srt_img[floor_y][floor_x + 1][1]) \
                                + v * (1-u) * int(srt_img[floor_y + 1][floor_x][1]) + u * v * int(srt_img[floor_y + 1][floor_x + 1][1])

                dst[i][j][2] = (1-u) * (1-v) * int(srt_img[floor_y][floor_x][2]) + u * (1 - v) * int(srt_img[floor_y][floor_x + 1][2]) \
                                + v * (1-u) * int(srt_img[floor_y + 1][floor_x][2]) + u * v * int(srt_img[floor_y + 1][floor_x + 1][2])
                
        return dst
               
#   81 landmarks
    def mean_face(self):
        np_mean_face_0_2 = np.loadtxt('/data/DLFS/DLFS-main/datasets/males/train0-2/landmarks_81/mean_landmarks/train0-2.txt', dtype=int)
        np_mean_face_3_6 = np.loadtxt('/data/DLFS/DLFS-main/datasets/males/train3-6/landmarks_81/mean_landmarks/train3-6.txt', dtype=int)
        np_mean_face_7_9 = np.loadtxt('/data/DLFS/DLFS-main/datasets/males/train7-9/landmarks_81/mean_landmarks/train7-9.txt', dtype=int)
        np_mean_face_15_19 = np.loadtxt('/data/DLFS/DLFS-main/datasets/males/train15-19/landmarks_81/mean_landmarks/train15-19.txt', dtype=int)
        np_mean_face_30_39 = np.loadtxt('/data/DLFS/DLFS-main/datasets/males/train30-39/landmarks_81/mean_landmarks/train30-39.txt', dtype=int)
        np_mean_face_50_69 = np.loadtxt('/data/DLFS/DLFS-main/datasets/males/train50-69/landmarks_81/mean_landmarks/train50-69.txt', dtype=int)
       
        self.mean_face_list = [np_mean_face_0_2, np_mean_face_3_6, np_mean_face_7_9, np_mean_face_15_19, np_mean_face_30_39, np_mean_face_50_69]





    def parallelize(self, model):
        # parallelize a network
        if self.isTrain and len(self.gpu_ids) > 0:
            return networks._CustomDataParallel(model)
        else:
            return model


    def requires_grad(self, model, flag=True):
        # freeze network weights
        for p in model.parameters():
            p.requires_grad = flag


    def accumulate(self, model1, model2, decay=0.999):
        # implements exponential moving average
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        model1_parallel = isinstance(model1, nn.DataParallel)
        model2_parallel = isinstance(model2, nn.DataParallel)

        for k in params1.keys():
            if model2_parallel and not model1_parallel:
                k2 = 'module.' + k
            elif model1_parallel and not model2_parallel:
                k2 = re.sub('module.', '', k)
            else:
                k2 = k
            params1[k].data.mul_(decay).add_(1 - decay, params2[k2].data)

    
    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty


    def set_inputs(self, data, mode='train'):
        # set input data to feed to the network
        if mode == 'train':
            real_A = data['A']
            real_B = data['B']

            # 1
            ref_A = data['A_ref']
            ref_B = data['B_ref']

            ref_A_1 = data['A_ref_1']
            ref_B_1 = data['B_ref_1']

            self.class_A = data['A_class']
            self.class_B = data['B_class']
            # print(self.class_B)  tensor([5])
            #self.class_distan = torch.tensor([1,2,3,0,5,4])

            self.reals = torch.cat((real_A, real_B), 0)
            # 1
            self.ref = torch.cat((ref_A, ref_B), 0)
            self.ref_1 = torch.cat((ref_A_1, ref_B_1), 0)
            
            # 1
            self.landmarks_A = data['A_landmarks'].cuda()
            self.landmarks_B = data['B_landmarks'].cuda()


            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()

                # 1
                self.ref = self.ref.cuda()
                self.ref_1 = self.ref_1.cuda()
                #self.reals_swap = self.reals_swap.cuda()

        else:
            self.img_paths = data['Paths']
            
            inputs = data['Imgs']
           
            # print(inputs.shape)   torch.Size([6, 3, 256, 256])  test时一次输入6张
            inputs_landmarks = data['landmarks']
            # print(inputs_landmarks.shape)  torch.Size([6, 32])

            if inputs.dim() > 4:
                inputs = inputs.squeeze(0)

            self.class_A = data['Classes']

            if self.class_A.dim() > 1:
                self.class_A = self.class_A.squeeze(0)

            if torch.is_tensor(data['Valid']):
                self.valid = data['Valid'].bool()
            else:
                self.valid = torch.ones(1, dtype=torch.bool)

            if self.valid.dim() > 1:
                self.valid = self.valid.squeeze(0)

            if isinstance(data['Paths'][0], tuple):
                self.image_paths = [path[0] for path in data['Paths']]
            else:
                self.image_paths = data['Paths']

            self.isEmpty = False if any(self.valid) else True  
            if not self.isEmpty:
                # available_idx = torch.arange(len(self.class_A))
                # select_idx = torch.masked_select(available_idx, self.valid).long()
                select_idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
                # print(select_idx)   #tensor([0, 1, 2, 3, 4, 5])
                # print(select_idx.dtype)
    
                inputs = torch.index_select(inputs, 0, select_idx)
                # print("index_select hou inputs size: ", inputs.shape)
                # print(inputs.shape)  torch.Size([6, 3, 256, 256])
    # 1
                inputs_landmarks = torch.index_select(inputs_landmarks, 0, select_idx)
                # print(inputs_landmarks.shape)  torch.Size([6, 32])

                self.class_A = torch.index_select(self.class_A, 0, select_idx)
            
#  ################################
            self.reals = inputs
            self.real_landmarks = inputs_landmarks
           

            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()
                # 1
                self.real_landmarks = self.real_landmarks.cuda()

    def get_z(self, mode='train'):
        self.z_0 = torch.randn((2, 16)).cuda(0)
        self.z_cyc = torch.randn((2, 16)).cuda(0)
        self.z_1 = torch.randn((2, 16)).cuda(0)

        if mode == 'test':
            self.z_0 = torch.randn((2, 16)).cuda(0)
            self.z_1 = torch.randn((2, 16)).cuda(0)
            self.z_2 = torch.randn((2, 16)).cuda(0)
            self.z_3 = torch.randn((2, 16)).cuda(0)



    def get_conditions(self, mode='train'):
        # set conditional inputs to the network
        if mode == 'train':
            nb = self.reals.shape[0] // 2
        elif self.traverse or self.deploy:
            if self.traverse and self.compare_to_trained_outputs:
                nb = 2
            else:
                nb = self.numClasses
        else:
            nb = self.numValid

        #tex condition mapping
        condG_A_gen = self.Tensor(nb, self.cond_length) # [1, 300]
        condG_B_gen = self.Tensor(nb, self.cond_length)
        condG_A_orig = self.Tensor(nb, self.cond_length)
        condG_B_orig = self.Tensor(nb, self.cond_length)
        #condG_A_distan = self.Tensor(nb, self.cond_length)
        #condG_B_distan = self.Tensor(nb, self.cond_length)

        # 1
        condG_A_gen1 = self.Tensor(nb, self.cond_length)
        condG_B_gen1 = self.Tensor(nb, self.cond_length)
        condG_A_orig1 = self.Tensor(nb, self.cond_length)
        condG_B_orig1 = self.Tensor(nb, self.cond_length)


        if self.no_cond_noise:
            noise_sigma = 0
        else:
            noise_sigma = 0.2

        for i in range(nb):
            condG_A_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
            condG_A_gen[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1

            # 1
            condG_A_gen1[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
            condG_A_gen1[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1

            if not (self.traverse or self.deploy):
                condG_B_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_gen[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_A_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_A_orig[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_B_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_orig[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1
                #1
                condG_B_gen1[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_gen1[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_A_orig1[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_A_orig1[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_B_orig1[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_orig1[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1


        if mode == 'train':
            self.gen_conditions = torch.cat((condG_A_gen, condG_B_gen), 0) #torch.cat((self.class_B, self.class_A), 0)
            # if the results are not good this might be the issue!!!! uncomment and update code respectively
            self.cyc_conditions = torch.cat((condG_B_gen, condG_A_gen), 0)
            self.orig_conditions = torch.cat((condG_A_orig, condG_B_orig), 0)
            #1
            self.gen_conditions1 = torch.cat((condG_A_gen1, condG_B_gen1), 0) #torch.cat((self.class_B, self.class_A), 0)
            # if the results are not good this might be the issue!!!! uncomment and update code respectively
            self.cyc_conditions1 = torch.cat((condG_B_gen1, condG_A_gen1), 0)
            self.orig_conditions1 = torch.cat((condG_A_orig1, condG_B_orig1), 0)

        else:
            self.gen_conditions = condG_A_gen #self.class_B
            self.gen_conditions1 = condG_A_gen1
            if not (self.traverse or self.deploy):
                # if the results are not good this might be the issue!!!! uncomment and update code respectively
                self.cyc_conditions = condG_B_gen #self.class_A
                self.orig_conditions = condG_A_orig

                self.cyc_conditions1 = condG_B_gen1  # self.class_A
                self.orig_conditions1 = condG_A_orig1


    def update_G(self, infer=False):
        # Generator optimization setp
        self.optimizer_G.zero_grad()
        self.get_conditions()

        ############### multi GPU ###############
        # 2
        rec_images, gen_images, cyc_images, orig_id_features, \
        orig_age_features, fake_id_features, fake_age_features = \
        self.netG(self.reals, self.gen_conditions, self.cyc_conditions, self.orig_conditions)


        # 2
        disc_out = self.netD(gen_images)

        # 2
        if self.opt.lambda_rec > 0:
            loss_G_Rec = self.criterionRec(rec_images, self.reals) * self.opt.lambda_rec
        else:
            loss_G_Rec = torch.zeros(1).cuda()

        # 2
        if self.opt.lambda_cyc > 0:
            loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc
        else:
            loss_G_Cycle = torch.zeros(1).cuda()

        # identity feature loss
        # 2
        loss_G_identity_reconst = self.identity_reconst_criterion(fake_id_features, orig_id_features) * self.opt.lambda_id
        loss_G_age_reconst = self.age_reconst_criterion(fake_age_features, self.gen_conditions) * self.opt.lambda_age
        loss_G_age_reconst += self.age_reconst_criterion(orig_age_features, self.orig_conditions) * self.opt.lambda_age

        # adversarial loss
        # 2
        target_classes = torch.cat((self.class_B,self.class_A),0)
        loss_G_GAN = self.criterionGAN(disc_out, target_classes, True, is_gen=True)

        # 2
        loss_G = (loss_G_GAN + loss_G_Rec + loss_G_Cycle + \
        loss_G_identity_reconst + loss_G_age_reconst).mean()
        loss_G.backward()
        self.optimizer_G.step()

        
        if self.use_moving_avg:
            self.accumulate(self.g_running, self.netG)
        

        # generate images for visdom
        if infer:
            if self.use_moving_avg:
                with torch.no_grad():
                    orig_id_features_out,_ = self.g_running.encode(self.reals)
                    #within domain decode
                    if self.opt.lambda_rec > 0:
                        rec_images_out = self.g_running.decode(orig_id_features_out, self.orig_conditions)

                    #cross domain decode
                    gen_images_out = self.g_running.decode(orig_id_features_out, self.gen_conditions)
                    #encode generated
                    fake_id_features_out,_ = self.g_running.encode(gen_images)
                    #decode generated
                    if self.opt.lambda_cyc > 0:
                        cyc_images_out = self.g_running.decode(fake_id_features_out, self.cyc_conditions)
            else:
                gen_images_out = gen_images
                if self.opt.lambda_rec > 0:
                    rec_images_out = rec_images
                if self.opt.lambda_cyc > 0:
                    cyc_images_out = cyc_images

        loss_dict = {'loss_G_Adv': loss_G_GAN.mean(), 'loss_G_Cycle': loss_G_Cycle.mean(),
                     'loss_G_Rec': loss_G_Rec.mean(), 'loss_G_identity_reconst': loss_G_identity_reconst.mean(),
                     'loss_G_age_reconst': loss_G_age_reconst.mean()}

        return [loss_dict,
                None if not infer else self.reals,
                None if not infer else gen_images_out,
                None if not infer else rec_images_out,
                None if not infer else cyc_images_out]
    


    def updata_M(self, infer=False):
        # 训练 Mapping network 
        mapping_input = torch.randn(2, 16).cuda()

        target_style = self.mapping_net(mapping_input, self.class_B, self.class_A)
        gen_images1, _, _, \
        _, _ , _, _, _, = self.netG(self.reals, target_style)

        s_pred_g = self.style_encoder(gen_images1, self.class_B, self.class_A)

        loss_sty = torch.mean(torch.abs(s_pred_g - target_style))

# adv
        disc_out = self.netD(gen_images1)  # 各年龄段的分数
        target_classes = torch.cat((self.class_B,self.class_A),0)
        loss_G_GAN_gen = self.criterionGAN(disc_out, target_classes, True, is_gen=True)
        loss_G_GAN = loss_G_GAN_gen 

# cycle image
        cyc_style = self.style_encoder(gen_images1, self.class_A, self.class_B)
        cyc_images, _, _, \
        _, _ , _, _, _, = self.netG(gen_images1, cyc_style)
        s_pred_c = self.style_encoder(cyc_images, self.class_A, self.class_B)
        loss_sty_c = torch.mean(torch.abs(s_pred_c - cyc_style))

        if self.opt.lambda_cyc > 0:
            # loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc + self.criterionCycle(cyc_images1, self.reals) * self.opt.lambda_cyc
            loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc
        else:
            loss_G_Cycle = torch.zeros(1).cuda()

        Loss1 = (loss_sty + loss_sty_c + loss_G_GAN + loss_G_Cycle).mean()


        self.optimizer_M.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_StyE.zero_grad()

        Loss1.backward()

        self.optimizer_M.step()
        self.optimizer_G.step()
        self.optimizer_StyE.step()
        
        
        if self.use_moving_avg:
            self.accumulate(self.g_running, self.netG)

        # generate images for visdom
        if infer:
            if self.use_moving_avg:
            
                with torch.no_grad():
                    target_style = self.mapping_net(mapping_input, self.class_B, self.class_A)
                    gen_images1, _, _, \
                     _, _ , _, _, _, = self.g_running(self.reals, target_style)

        loss_dict = {'loss_sty1': (loss_sty + loss_sty_c).mean()}

        return [loss_dict, 
                None if not infer else gen_images1]
    



    def update_distan_G_Map(self, infer=False, epoch=0):
        self.get_z()

        style_0 = self.mapping_net(self.z_0, self.class_B, self.class_A)
        # print(style_0.shape)
        # print(style_0.shape)  # torch.Size([2, 256])

        gen_images, _, _, \
        _, _ , _, _, _, = self.netG(self.reals, style_0)
        # print(gen_images.shape)  # torch.Size([2, 3, 256, 256])

        disc_out = self.netD(gen_images)  
        target_classes = torch.cat((self.class_B, self.class_A),0)
        loss_G_GAN_gen = self.criterionGAN(disc_out, target_classes, True, is_gen=True)
        loss_G_GAN = loss_G_GAN_gen 

        s_pred_g = self.style_encoder(gen_images, self.class_B, self.class_A)
        # 1
        # loss_sty = torch.mean(torch.abs(s_pred_g - style_0))*10
        # loss_sty = torch.mean(torch.abs(s_pred_g - style_0))*100


        ori_style = self.style_encoder(self.reals, self.class_A, self.class_B)
        # 1 ? 
        # cyc_style = self.style_encoder(gen_images, self.class_A, self.class_B)

# 1  ? cyc
        # cyc_images, _, _, \
        # _, _ , _, _, _, = self.netG(gen_images, cyc_style)
        cyc_images, _, _, \
        _, _ , _, _, _, = self.netG(gen_images, ori_style)
     



        loss_sty = torch.mean(torch.abs(s_pred_g - style_0))*10 

        # self-reconstruction loss
        if self.opt.lambda_rec > 0:
            # loss_G_Rec = self.criterionRec(rec_images, self.reals) * self.opt.lambda_rec + self.criterionRec(rec_images1, self.reals) * self.opt.lambda_rec
            rec_images, _, _, \
            _, _ , _, _, _, = self.netG(self.reals, ori_style)
            s_pred_r = self.style_encoder(rec_images, self.class_A, self.class_B)

            loss_G_Rec = self.criterionRec(rec_images, self.reals) * self.opt.lambda_rec
            # print(loss_G_Rec.dtype)  torch.float32
        else:
            loss_G_Rec = torch.zeros(1).cuda()

        #cycle loss
        if self.opt.lambda_cyc > 0:
            # loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc + self.criterionCycle(cyc_images1, self.reals) * self.opt.lambda_cyc
            loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc
        else:
            loss_G_Cycle = torch.zeros(1).cuda()


        style_1 = self.mapping_net(self.z_1, self.class_B, self.class_A)
        gen_images1, _, _, \
        _, _ , _, _, _, = self.netG(self.reals, style_1)

   
 
        loss_ds_0 = torch.mean(torch.abs(gen_images[0] - gen_images1[0]))*(0.6)
        loss_ds_1 = torch.mean(torch.abs(gen_images[1] - gen_images1[1]))*(0.6)

        rgb_ds = loss_ds_0
  
        if loss_ds_0 > (0.07):
            loss_ds_0 = 0.07
            loss_ds_0 = torch.tensor(loss_ds_0, dtype=torch.float32).cuda()
        if loss_ds_1 > (0.07):
            loss_ds_1 = 0.07
            loss_ds_1 = torch.tensor(loss_ds_1, dtype=torch.float32).cuda()
        

        loss_Per = self.PerceptualLoss(cyc_images, self.reals) 
        # loss_Per = torch.zeros(1).cuda()

# #################################
# 1  加入人种约束
        # start = time.time()
        loss_G_race = torch.zeros(1).cuda()
        loss_G_arc = torch.zeros(1).cuda()
        gen_race_index = -1
        input_race_index = -1
        if epoch > (3):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
            trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
            gen_images0 = gen_images[0]
            gen_images0_numpy = gen_images0.detach().cpu().float().numpy()
            gen_images0_numpy = (np.transpose(gen_images0_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            gen_images0_numpy = gen_images0_numpy.astype(np.uint8)
            gen_images0_numpy_copy = gen_images0_numpy

            input_images0 = self.reals[0]
            input_images0_numpy = input_images0.detach().cpu().float().numpy()
            input_images0_numpy = (np.transpose(input_images0_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            input_images0_numpy = input_images0_numpy.astype(np.uint8)
            input_images0_numpy_copy = input_images0_numpy
        

      
            image0 = trans(gen_images0_numpy)
            image0 = image0.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            image0 = image0.to(device)

 
            self.model_fair_4.eval()
 
            outputs_gen = self.model_fair_4(image0)
            outputs_race = outputs_gen[0][:4]  
            outputs_race = torch.exp(outputs_race) / torch.sum(torch.exp(outputs_race))
            gen_race_index = torch.argmax(outputs_race)

         
            input0 = trans(input_images0_numpy)
            input0 = input0.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            input0 = input0.to(device)

        
            self.model_fair_4.eval()
            outputs_input = self.model_fair_4(input0)
            outputs_race_input = outputs_input[0][:4]
            outputs_race_input = torch.exp(outputs_race_input) / torch.sum(torch.exp(outputs_race_input))
            input_race_index = torch.argmax(outputs_race_input)

        
            if gen_race_index == input_race_index:
                loss_G_race = torch.zeros(1).cuda()
      
            else:
                loss_G_race = torch.ones(1).cuda() - outputs_race[input_race_index]
            

            try:
                self.arc_model.eval()
                gen_images0_numpy_copy = cv2.cvtColor(gen_images0_numpy_copy, cv2.COLOR_RGB2BGR)
                faces_gen = self.face_engine.detectMultiScale(gen_images0_numpy_copy, scaleFactor=1.3, minNeighbors=5)
                if(len(faces_gen) != 0):
                    for (x, y, w, h) in faces_gen:
                        gen_images0_numpy_copy = gen_images0_numpy_copy[y:y+h, x:x+w]
                        gen_images0_numpy_copy = cv2.resize(gen_images0_numpy_copy, (128,128))
                
                    
                    input_images0_numpy_copy = cv2.cvtColor(input_images0_numpy_copy, cv2.COLOR_RGB2BGR)
                    faces = self.face_engine.detectMultiScale(input_images0_numpy_copy, scaleFactor=1.3, minNeighbors=5)
                    if(len(faces) != 0):   
                        for (x, y, w, h) in faces:
                      
                            input_images0_numpy_copy = input_images0_numpy_copy[y:y+h, x:x+w]
                            input_images0_numpy_copy = cv2.resize(input_images0_numpy_copy, (128,128))
                        
                        img_gen = cv2.cvtColor(gen_images0_numpy_copy, cv2.COLOR_BGR2GRAY)
                        img_gen = np.dstack((img_gen, np.fliplr(img_gen)))
                        img_gen = img_gen.transpose((2, 0, 1))
                        img_gen = img_gen[:, np.newaxis, :, :]
                        img_gen = img_gen.astype(np.float32, copy=False)
                        img_gen -= 127.5
                        img_gen /= 127.5

                        img_input = cv2.cvtColor(input_images0_numpy_copy, cv2.COLOR_BGR2GRAY)
                        img_input = np.dstack((img_input, np.fliplr(img_input)))
                        img_input = img_input.transpose((2, 0, 1))
                        img_input = img_input[:, np.newaxis, :, :]
                        img_input = img_input.astype(np.float32, copy=False)
                        img_input -= 127.5
                        img_input /= 127.5

                        img_gen = torch.from_numpy(img_gen)
                        img_gen = img_gen.to(torch.device("cuda"))
                        output_gen = self.arc_model(img_gen)
                        output_gen = output_gen.data.cpu().numpy()
                        gen_fe_1 = output_gen[::2]
                        gen_fe_2 = output_gen[1::2]
                        gen_feature = np.hstack((gen_fe_1, gen_fe_2))

                        img_input = torch.from_numpy(img_input)
                        img_input = img_input.to(torch.device("cuda"))
                        output_real = self.arc_model(img_input)
                        output_real = output_real.data.cpu().numpy()
                        real_fe_1 = output_real[::2]
                        real_fe_2 = output_real[1::2]
                        real_feature = np.hstack((real_fe_1, real_fe_2))

                        loss_G_arc = np.dot(gen_feature[0], real_feature[0]) / (np.linalg.norm(gen_feature[0]) * np.linalg.norm(real_feature[0]))
                        loss_G_arc = (torch.tensor(loss_G_arc, dtype=torch.float32)).cuda()
                        loss_G_arc = (torch.ones(1).cuda() - loss_G_arc) * (0.2)
            except:
                loss_G_arc = torch.zeros(1).cuda()
                print('arcface false')
                    
       

        # 1 overall loss
        #  - loss_ds
        loss_G = (loss_G_GAN + loss_G_Rec + loss_G_Cycle + loss_sty - loss_ds_0 - loss_ds_1 + loss_Per + loss_G_race + loss_G_arc).mean()
        # loss_G = (loss_G_GAN + loss_G_Cycle + loss_sty + loss_G_Rec ).mean()
       
       

        self.optimizer_G.zero_grad()
        self.optimizer_StyE.zero_grad()
        self.optimizer_M.zero_grad()

        loss_G.backward()

        self.optimizer_G.step()
        self.optimizer_StyE.step()
        self.optimizer_M.step()

         # generate images for visdom
        if infer:
        # if True:
            if self.use_moving_avg:
                # with open('latent.txt', 'ab') as f:
                #     np.savetxt(f, ori_latent.cpu().detach().numpy(), fmt='%.3e', header='ori_latent start')
                #     np.savetxt(f, ori_latent1.cpu().detach().numpy(), fmt='%.3e', header='ori_latent1 start', footer='----------')
               
                with torch.no_grad():

                     # 1
                    gen_images_ori, _, _, \
                     _, _ , _, _, _, = self.g_running(self.reals, style_0)
                    gen_images_ori = gen_images_ori
                   

                    gen_images1_ori, _, _, \
                     _, _ , _, _, _, = self.g_running(self.reals, style_1)
                    gen_images_ori1 = gen_images1_ori



    
                    z_many = torch.randn(1000, 16).cuda(0)
                    # class_B = torch.LongTensor(10000).cuda(0).fill_(self.class_B[0])
                    # class_A = torch.LongTensor(10000).cuda(0).fill_(self.class_A[0])
                    style_list = self.mapping_net(z_many, self.class_B, self.class_A, infer=True)
         
                    a_style = torch.squeeze(style_list[0])
                    b_style = torch.squeeze(style_list[1])
                    a_avg = torch.mean(a_style, dim=0, keepdim=True)
                    b_avg = torch.mean(b_style, dim=0, keepdim=True)
                    # print(b_avg.shape)  # torch.Size([1, 256])
                    style_avg = torch.cat((a_avg, b_avg))   
                    # print(style_avg.shape)# torch.Size([2, 256])
                    style_0 = torch.lerp(style_avg, style_0, 0.5) 
                    style_1 = torch.lerp(style_avg, style_1, 0.5)
    
                    # 、、、、、、、、、、、、

                    # 1
                    gen_images, _, _, \
                     _, _ , _, _, _, = self.g_running(self.reals, style_0)

                 
                    gen_images_out = gen_images
                   

                    # 1
                    gen_images1, _, _, \
                     _, _ , _, _, _, = self.g_running(self.reals, style_1)
                    gen_images_out1 = gen_images1

                    #encode generated
                    orig_id_features_out, orig_struct, orig_text, _ = self.g_running.encode(self.reals)
                    fake_id_features_out, fake_struct, fake_text, _ = self.g_running.encode(gen_images)
                    ori_style = self.style_encoder(self.reals, self.class_A, self.class_B)
                    # cyc_style = self.style_encoder(gen_images, self.class_A, self.class_B)
                    
                    #decode generated
                    if self.opt.lambda_cyc > 0:
                        # ??? cyc style
                        # cyc_images_out = self.g_running.decode(fake_struct, fake_text, latent=cyc_style)
                        cyc_images_out = self.g_running.decode(fake_struct, fake_text, latent=ori_style)
                    if self.opt.lambda_rec > 0:
                        rec_images_out = self.g_running.decode(orig_struct, orig_text, latent=ori_style)
            else:
                gen_images_out = gen_images
                # 1
                gen_images_out1 = gen_images1

                if self.opt.lambda_rec > 0:
                    rec_images_out = rec_images
                if self.opt.lambda_cyc > 0:
                    cyc_images_out = cyc_images

        # 1
        loss_dict = {'loss_G_arc': loss_G_arc, 'loss_Per': loss_Per.mean(), 'loss_G_div': loss_ds_0, 'loss_G_div1': loss_ds_1, 'rgb_ds': rgb_ds, 'loss_G_gen_Adv': loss_G_GAN_gen.mean(), 'loss_sty': loss_sty.mean(),
                     'loss_G_Rec': loss_G_Rec.mean(), 'loss_G_Cycle': loss_G_Cycle.mean(), 'loss_G_race':loss_G_race, 'in_race': input_race_index, 'gen_race':gen_race_index}


        return [loss_dict,
                None if not infer else self.reals,
                None if not infer else gen_images_out,
                None if not infer else gen_images_out1,
                # None if not infer else rec_images_out,
                None if not infer else None,
                None if not infer else cyc_images_out,
                None if not infer else gen_images_ori,
                None if not infer else gen_images_ori1,]
                # None if not infer else new_landmarks_A_recover,
                # None if not infer else new_landmarks_B_recover]

        



    def update_distan_G(self, infer=False, epoch=0):
     

        target_style = self.style_encoder(self.ref, self.class_B, self.class_A)
        
      
        gen_images, _, _, \
        _, _ , _, _, _, = self.netG(self.reals, target_style)

        s_pred_g = self.style_encoder(gen_images, self.class_B, self.class_A)
        # loss_sty = torch.mean(torch.abs(s_pred_g - target_style))*10

 
        ori_style = self.style_encoder(self.reals, self.class_A, self.class_B)
        # cyc_style = self.style_encoder(gen_images, self.class_A, self.class_B)

       
        cyc_images, _, _, \
        _, _ , _, _, _, = self.netG(gen_images, ori_style)
        # s_pred_c = self.style_encoder(cyc_images, self.class_A, self.class_B)




        loss_sty = torch.mean(torch.abs(s_pred_g - target_style))*10 


      
        target_style_1 = self.style_encoder(self.ref_1, self.class_B, self.class_A)
        gen_images_1, _, _, \
        _, _ , _, _, _, = self.netG(self.reals, target_style_1)


        #discriminator pass
        # print(gen_images.dtype)
        disc_out = self.netD(gen_images)  
        # print(disc_out.shape)  #torch.Size([2, 6, 1, 1])
         # adversarial loss
        target_classes = torch.cat((self.class_B,self.class_A),0)

        loss_G_GAN_gen = self.criterionGAN(disc_out, target_classes, True, is_gen=True)
    
        loss_G_GAN = loss_G_GAN_gen 

        # 1
        loss_ds_0 = torch.mean(torch.abs(gen_images[0] - gen_images_1[0]))*(0.6)
        loss_ds_1 = torch.mean(torch.abs(gen_images[1] - gen_images_1[1]))*(0.6)

        if loss_ds_0 > (0.07):
            loss_ds_0 = 0.07
            loss_ds_0 = torch.tensor(loss_ds_0, dtype=torch.float32).cuda()
        if loss_ds_1 > (0.07):
            loss_ds_1 = 0.07
            loss_ds_1 = torch.tensor(loss_ds_1, dtype=torch.float32).cuda()

     

        loss_Per = self.PerceptualLoss(cyc_images, self.reals) 
        # loss_Per = torch.zeros(1).cuda()
        
        # self-reconstruction loss
        if self.opt.lambda_rec > 0:
            # loss_G_Rec = self.criterionRec(rec_images, self.reals) * self.opt.lambda_rec + self.criterionRec(rec_images1, self.reals) * self.opt.lambda_rec
            rec_images, _, _, \
            _, _ , _, _, _, = self.netG(self.reals, ori_style)
            s_pred_r = self.style_encoder(rec_images, self.class_A, self.class_B)

            loss_G_Rec = self.criterionRec(rec_images, self.reals) * self.opt.lambda_rec
        else:
            loss_G_Rec = torch.zeros(1).cuda()

        #cycle loss
        if self.opt.lambda_cyc > 0:
            # loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc + self.criterionCycle(cyc_images1, self.reals) * self.opt.lambda_cyc
            loss_G_Cycle = self.criterionCycle(cyc_images, self.reals) * self.opt.lambda_cyc
        else:
            loss_G_Cycle = torch.zeros(1).cuda()


        loss_G_race = torch.zeros(1).cuda()

        loss_G_arc = torch.zeros(1).cuda()
        if epoch > (3):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


            trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
            gen_images0 = gen_images[0]
            gen_images0_numpy = gen_images0.detach().cpu().float().numpy()
            gen_images0_numpy = (np.transpose(gen_images0_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            gen_images0_numpy = gen_images0_numpy.astype(np.uint8)
            gen_images0_numpy_copy = gen_images0_numpy

            input_images0 = self.reals[0]
            input_images0_numpy = input_images0.detach().cpu().float().numpy()
            input_images0_numpy = (np.transpose(input_images0_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            input_images0_numpy = input_images0_numpy.astype(np.uint8)
            input_images0_numpy_copy = input_images0_numpy
        

     
            image0 = trans(gen_images0_numpy)
            image0 = image0.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            image0 = image0.to(device)


            self.model_fair_4.eval()
          
            outputs_gen = self.model_fair_4(image0)
            outputs_race = outputs_gen[0][:4]  
            outputs_race = torch.exp(outputs_race) / torch.sum(torch.exp(outputs_race))
            gen_race_index = torch.argmax(outputs_race)

  
            input0 = trans(input_images0_numpy)
            input0 = input0.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            input0 = input0.to(device)


            self.model_fair_4.eval()
            outputs_input = self.model_fair_4(input0)
            outputs_race_input = outputs_input[0][:4]
            outputs_race_input = torch.exp(outputs_race_input) / torch.sum(torch.exp(outputs_race_input))
            input_race_index = torch.argmax(outputs_race_input)

         
            if gen_race_index == input_race_index:
                loss_G_race = torch.zeros(1).cuda()
          
            else:
                loss_G_race = torch.ones(1).cuda() - outputs_race[input_race_index]


            try:
                self.arc_model.eval()
                gen_images0_numpy_copy = cv2.cvtColor(gen_images0_numpy_copy, cv2.COLOR_RGB2BGR)
                faces_gen = self.face_engine.detectMultiScale(gen_images0_numpy_copy, scaleFactor=1.3, minNeighbors=5)
                if(len(faces_gen) != 0):
                    for (x, y, w, h) in faces_gen:
                        gen_images0_numpy_copy = gen_images0_numpy_copy[y:y+h, x:x+w]
                        gen_images0_numpy_copy = cv2.resize(gen_images0_numpy_copy, (128,128))
                
                   
                    input_images0_numpy_copy = cv2.cvtColor(input_images0_numpy_copy, cv2.COLOR_RGB2BGR)
                    faces = self.face_engine.detectMultiScale(input_images0_numpy_copy, scaleFactor=1.3, minNeighbors=5)
                    if(len(faces) != 0):   
                        for (x, y, w, h) in faces:
                    
                            input_images0_numpy_copy = input_images0_numpy_copy[y:y+h, x:x+w]
                            input_images0_numpy_copy = cv2.resize(input_images0_numpy_copy, (128,128))
                        
                        img_gen = cv2.cvtColor(gen_images0_numpy_copy, cv2.COLOR_BGR2GRAY)
                        img_gen = np.dstack((img_gen, np.fliplr(img_gen)))
                        img_gen = img_gen.transpose((2, 0, 1))
                        img_gen = img_gen[:, np.newaxis, :, :]
                        img_gen = img_gen.astype(np.float32, copy=False)
                        img_gen -= 127.5
                        img_gen /= 127.5

                        img_input = cv2.cvtColor(input_images0_numpy_copy, cv2.COLOR_BGR2GRAY)
                        img_input = np.dstack((img_input, np.fliplr(img_input)))
                        img_input = img_input.transpose((2, 0, 1))
                        img_input = img_input[:, np.newaxis, :, :]
                        img_input = img_input.astype(np.float32, copy=False)
                        img_input -= 127.5
                        img_input /= 127.5

                        img_gen = torch.from_numpy(img_gen)
                        img_gen = img_gen.to(torch.device("cuda"))
                        output_gen = self.arc_model(img_gen)
                        output_gen = output_gen.data.cpu().numpy()
                        gen_fe_1 = output_gen[::2]
                        gen_fe_2 = output_gen[1::2]
                        gen_feature = np.hstack((gen_fe_1, gen_fe_2))

                        img_input = torch.from_numpy(img_input)
                        img_input = img_input.to(torch.device("cuda"))
                        output_real = self.arc_model(img_input)
                        output_real = output_real.data.cpu().numpy()
                        real_fe_1 = output_real[::2]
                        real_fe_2 = output_real[1::2]
                        real_feature = np.hstack((real_fe_1, real_fe_2))

                        loss_G_arc = np.dot(gen_feature[0], real_feature[0]) / (np.linalg.norm(gen_feature[0]) * np.linalg.norm(real_feature[0]))
                        loss_G_arc = (torch.tensor(loss_G_arc, dtype=torch.float32)).cuda()
                        loss_G_arc = (torch.ones(1).cuda() - loss_G_arc) * (0.2)
            except:
                loss_G_arc = torch.zeros(1).cuda()
                print('arcface false')

            

        # 1 overall loss
        loss_G = (loss_G_GAN + loss_G_Rec + loss_G_Cycle + loss_sty - loss_ds_0 - loss_ds_1 + loss_Per + loss_G_race + loss_G_arc).mean()
        # loss_G = (loss_G_GAN + loss_G_Cycle + loss_sty + loss_G_Rec ).mean()

        # self.optimizer_L.zero_grad()
        self.optimizer_G.zero_grad()
        # self.optimizer_StyE.zero_grad()
        # self.optimizer_StyE.zero_grad()

        loss_G.backward()

        # self.optimizer_L.step()
        self.optimizer_G.step()
        # self.optimizer_StyE.step()
  

        # update exponential moving average
        if self.use_moving_avg:
            self.accumulate(self.g_running, self.netG)


        # generate images for visdom
     
        # 1
        loss_dict = {'loss_G_gen_Adv': loss_G_GAN_gen.mean(), 'loss_sty': loss_sty.mean(),
                     'loss_G_Rec': loss_G_Rec.mean(), 'loss_G_Cycle': loss_G_Cycle.mean()}


        return [loss_dict,]
         


    def update_D(self):
        # Discriminator optimization setp
        #import ipdb; ipdb.set_trace()
        self.optimizer_D.zero_grad()
        self.get_conditions()

        ############### multi GPU ###############
        _, gen_images, _, _, _, _, _ = self.netG(self.reals, self.gen_conditions, None, None, disc_pass=True)

        #fake discriminator pass
        fake_disc_in = gen_images.detach()
        fake_disc_out = self.netD(fake_disc_in)

        #real discriminator pass
        real_disc_in = self.reals

        # necessary for R1 regularization
        real_disc_in.requires_grad_()

        real_disc_out = self.netD(real_disc_in)

        #Fake GAN loss
        fake_target_classes = torch.cat((self.class_B,self.class_A),0)
        loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)

        #Real GAN loss
        real_target_classes = torch.cat((self.class_A,self.class_B),0)
        loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)

        # R1 regularization
        loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)

        loss_D = (loss_D_fake + loss_D_real + loss_D_reg).mean()
        loss_D.backward()
        self.optimizer_D.step()

        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 'loss_D_reg': loss_D_reg.mean()}

    def update_distan_D_Map(self):
        
        self.get_z()

        with torch.no_grad():
            style_0 = self.mapping_net(self.z_0, self.class_B, self.class_A)

            gen_images, _, _, \
            _, _ , _, _, _, = self.netG(self.reals, style_0)
            # print(gen_images.shape)  # torch.Size([2, 3, 256, 256])

        fake_disc_out = self.netD(gen_images) 

        real_disc_in = self.reals
        real_disc_in.requires_grad_()
        real_disc_out = self.netD(real_disc_in)

        fake_target_classes = torch.cat((self.class_B,self.class_A),0)
        real_target_classes = torch.cat((self.class_A,self.class_B),0)

        loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)
        loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)
        loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)

        loss_D = (loss_D_fake + loss_D_real + loss_D_reg).mean() 

        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 'loss_D_reg': loss_D_reg.mean()}
        
    
    def update_distan_D(self):
    

        ############### multi GPU ###############
        with torch.no_grad():
            # target_style = self.style_encoder(self.ref, self.class_B, self.class_A)
        
            target_style = self.style_encoder(self.reals, self.class_A, self.class_B)

            gen_images, _, _, \
            _, _ , _, _, _, = self.netG(self.reals, target_style)
      
        # 1
        # fake_disc_in = gen_images.detach()
        fake_disc_in = gen_images
        

        #rec_disc_in = rec_images.detach()
        #swap_disc_in = swap_images.detach()
        fake_disc_out = self.netD(fake_disc_in)
        # print(fake_disc_out.shape) [2, 6, 1, 1]
        

        #real discriminator pass
        real_disc_in = self.reals

        # necessary for R1 regularization
        real_disc_in.requires_grad_()

        real_disc_out = self.netD(real_disc_in)

        #Fake GAN loss
        fake_target_classes = torch.cat((self.class_B,self.class_A),0)
        real_target_classes = torch.cat((self.class_A,self.class_B),0)
        # print(fake_target_classes)  #          content: tensor[3, 2]
        loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)
        #loss_D_rec = self.criterionGAN(rec_disc_out, real_target_classes, False, is_gen=False)
        #loss_D_swap = self.criterionGAN(swap_disc_out, fake_target_classes, False, is_gen=False)

        #Real GAN loss
        #real_target_classes = torch.cat((self.class_A,self.class_B),0)
        loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)

    
        loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)
     

        loss_D = (loss_D_fake + loss_D_real + loss_D_reg).mean() #+ cooccur_loss

        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 'loss_D_reg': loss_D_reg.mean()}

    def inference(self, data, landmarks_save_dir, select_two_age):

        self.landmarks_save_dir = landmarks_save_dir
        #import ipdb; ipdb.set_trace() 
        # 分 train  /   test
        self.set_inputs(data, mode='test')
        if self.isEmpty:
            return
        # print(self.class_A)  tensor([0, 1, 2, 3, 4, 5])
       
        self.reals = self.reals[select_two_age:select_two_age+1,:,:,:] 
        self.real_landmarks = self.real_landmarks[select_two_age:select_two_age+1, :]  

        #  /////////////////////////
        self.image_paths = self.image_paths[select_two_age]
        # //////////////////////////
        # print(self.image_paths)

        if(self.image_paths != ''):

            self.numValid = self.valid.sum().item()
            sz = self.reals.size()
            # print(sz)  torch.Size([6, 3, 256, 256])  

     
            self.fake_B = self.Tensor(self.numClasses, 1, sz[1], sz[2], sz[3])
            self.cyc_A = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            self.offset_landmarks_B = self.Tensor(self.numClasses, 1, 32)
            # 1
            # self.fake_B1 = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            self.fake_B1 = self.Tensor(self.numClasses, 1, sz[1], sz[2], sz[3])
            self.fake_B2 = self.Tensor(self.numClasses, 1, sz[1], sz[2], sz[3])
            self.fake_B3 = self.Tensor(self.numClasses, 1, sz[1], sz[2], sz[3])
    
            self.offset_landmarks_B1 = self.Tensor(self.numClasses, 1, 32)
            self.offset_landmarks_B2 = self.Tensor(self.numClasses, 1, 32)
            self.offset_landmarks_B3 = self.Tensor(self.numClasses, 1, 32)
            self.offset_landmarks_B4 = self.Tensor(self.numClasses, 1, 32)

            with torch.no_grad():
                if self.traverse or self.deploy:
                    if self.traverse and self.compare_to_trained_outputs:
                        start = self.compare_to_trained_class - self.trained_class_jump
                        end = start + (self.trained_class_jump * 2) * 2 #arange is between [start, end), end is always omitted
                        self.class_B = torch.arange(start, end, step=self.trained_class_jump*2, dtype=self.class_A.dtype)
                    else:
                        self.class_B = torch.arange(self.numClasses, dtype=self.class_A.dtype)

                    self.get_conditions(mode='test')

 
                    self.fake_B = self.netG.infer(self.reals, self.gen_conditions, traverse=self.traverse, deploy=self.deploy, interp_step=self.opt.interp_step)
                    self.fake_B1 = self.netG.infer(self.reals, self.gen_conditions1, traverse=self.traverse, deploy=self.deploy, interp_step=self.opt.interp_step)
                else:
                    # run
                    for i in range(self.numClasses):   
                    
      
                        self.class_B = self.Tensor(1).long().fill_(i)
                        # print(self.class_B)
                        # print(self.class_B.dtype)  torch.int64

                      
                        z_many = torch.randn(10000, 16).cuda(0)
                        # z_many = torch.randn(100, 16).cuda(0)

                        style_list = self.mapping_net(z_many, self.class_B, self.class_B, infer=True)
                        a_style = torch.squeeze(style_list[0])
                        b_style = torch.squeeze(style_list[1])
                        a_avg = torch.mean(a_style, dim=0, keepdim=True)
                        b_avg = torch.mean(b_style, dim=0, keepdim=True)
                        style_avg = torch.cat((a_avg, b_avg))   
                    
                    #  ######################################################


                        self.get_z(mode='test')
                      
                        style_0 = self.mapping_net(self.z_0, self.class_B, self.class_B)[0:1]
                        style_1 = self.mapping_net(self.z_1, self.class_B, self.class_B)[0:1]
                        style_2 = style_avg[0:1]  # 使用 平均 latent code
                        style_3 = self.mapping_net(self.z_3, self.class_B, self.class_B)[0:1]
                      
                

                        z_many_land = torch.randn(10000, 16).cuda(0)
                        # z_many = torch.randn(100, 16).cuda(0)
                        style_list_land = self.mapping_net_L(z_many_land, self.class_B, self.class_B, infer=True)
                        # print(style_list[0].shape)   #torch.Size([10000, 1, 64])
                        a_style_land = torch.squeeze(style_list_land[0])
                        b_style_land = torch.squeeze(style_list_land[1])

                        a_avg_land = torch.mean(a_style_land, dim=0, keepdim=True)
                        b_avg_land = torch.mean(b_style_land, dim=0, keepdim=True)
                        style_avg_land = torch.cat((a_avg_land, b_avg_land))   #  torch.Size([2, 64])
                    #  #######################################################################


                       
                        style_0_L = self.mapping_net_L(self.z_0, self.class_B, self.class_B)  # torch.Size([2, 64])
                        style_1_L = self.mapping_net_L(self.z_1, self.class_B, self.class_B)
                        style_2_L = style_avg_land  
                        # style_2_L = style_0_L
                # ###############################################################################
        
                        # style_0_L = torch.lerp(style_avg_land, style_0_L, 0.5) 
                        # style_1_L = torch.lerp(style_avg_land, style_1_L, 0.5) 
                        # style_2_L = self.mapping_net_L(self.z_2, self.class_B, self.class_B)


                        if self.isTrain:
                            self.fake_B[i, :, :, :, :] = self.g_running.infer(self.reals, self.gen_conditions)
                        else:
              
         
                            self.fake_B[i, :, :, :, :] = self.netG.infer(self.reals, style_0)
                            self.fake_B1[i, :, :, :, :] = self.netG.infer(self.reals, style_1)
                            self.fake_B2[i, :, :, :, :] = self.netG.infer(self.reals, style_2)
                            self.fake_B3[i, :, :, :, :] = self.netG.infer(self.reals, style_3)
          
                            self.offset_landmarks_B[i, :, :] = self.netLandG(self.real_landmarks, style_0_L[0])
                            self.offset_landmarks_B1[i, :, :] = self.netLandG(self.real_landmarks, style_1_L[1])
                            self.offset_landmarks_B2[i, :, :] = self.netLandG(self.real_landmarks, style_1_L[0])
                            self.offset_landmarks_B3[i, :, :] = self.netLandG(self.real_landmarks, style_2_L[0])
                            self.offset_landmarks_B4[i, :, :] = self.netLandG(self.real_landmarks, style_2_L[1])
                           
                        cyc_input = self.fake_B[i, :, :, :, :]

                        
    
                # 1
                self.get_landmarks()
                visuals = self.get_visuals()

            return visuals

        else:
            print('over')


    def save(self, which_epoch):
       
        self.save_network(self.mapping_net, 'Mapping', which_epoch, self.gpu_ids)
        self.save_network(self.style_encoder, 'Style_encoder', which_epoch, self.gpu_ids)


        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.use_moving_avg:
            self.save_network(self.g_running, 'g_running', which_epoch, self.gpu_ids)


    def update_learning_rate(self):
        lr = self.old_lr * self.opt.decay_gamma
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            mult = param_group.get('mult', 1.0)
            param_group['lr'] = lr * mult
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

# 1
    def get_landmarks(self):
        
        current_landmarks_save_dir = os.path.join(self.landmarks_save_dir, 'landmarks')
        if not os.path.exists(current_landmarks_save_dir):
            os.makedirs(current_landmarks_save_dir)

        img_landmarks_save_dir = os.path.join(self.landmarks_save_dir, 'images')
        if not os.path.exists(img_landmarks_save_dir):
            os.makedirs(img_landmarks_save_dir)
       
        for i in range(self.numClasses):
         
            current_landmarks = self.real_landmarks + self.offset_landmarks_B[i]
            # print(self.real_landmarks.shape)   #torch.Size([2, 32])
            current_landmarks = torch.mm(current_landmarks, torch.t(self.PCA)) + self.mean
            # print(self.real_landmarks.shape) torch.Size([2, 136])
            current_landmarks = current_landmarks.detach().cpu().numpy() * 256  
            current_landmarks = current_landmarks.reshape((1, 81, 2))  # 81 landmarks

            current_landmarks_1 = self.real_landmarks + self.offset_landmarks_B1[i]
            current_landmarks_1 = torch.mm(current_landmarks_1, torch.t(self.PCA)) + self.mean
            current_landmarks_1 = current_landmarks_1.detach().cpu().numpy() * 256  
            current_landmarks_1 = current_landmarks_1.reshape((1, 81, 2))

            current_landmarks_2 = self.real_landmarks + self.offset_landmarks_B2[i]
            current_landmarks_2 = torch.mm(current_landmarks_2, torch.t(self.PCA)) + self.mean
            current_landmarks_2 = current_landmarks_2.detach().cpu().numpy() * 256  
            current_landmarks_2 = current_landmarks_2.reshape((1, 81, 2))     

            current_landmarks_3 = self.real_landmarks + self.offset_landmarks_B3[i]
            current_landmarks_3 = torch.mm(current_landmarks_3, torch.t(self.PCA)) + self.mean
            current_landmarks_3 = current_landmarks_3.detach().cpu().numpy() * 256  
            current_landmarks_3 = current_landmarks_3.reshape((1, 81, 2))  
            
            current_landmarks_4 = self.real_landmarks + self.offset_landmarks_B4[i]
            current_landmarks_4 = torch.mm(current_landmarks_4, torch.t(self.PCA)) + self.mean
            current_landmarks_4 = current_landmarks_4.detach().cpu().numpy() * 256  
            current_landmarks_4 = current_landmarks_4.reshape((1, 81, 2)) 

          
            for m in range(1):  # 2 
              
                img_name = self.image_paths.split('/')[-1]
                img_name = img_name.split('.')[0] 
           
                txt_name = img_name + '_trans_to_class_' + str(i) + '_0' + '.txt'
                txt_name1 = img_name + '_trans_to_class_' + str(i) + '_1' + '.txt'
                txt_name2 = img_name + '_trans_to_class_' + str(i) + '_2' + '.txt'
                txt_name3 = img_name + '_trans_to_class_' + str(i) + '_3' + '.txt'
                txt_name4 = img_name + '_trans_to_class_' + str(i) + '_4' + '.txt'

         
                np.savetxt(os.path.join(current_landmarks_save_dir,txt_name), current_landmarks[m], fmt='%d')
                np.savetxt(os.path.join(current_landmarks_save_dir,txt_name1), current_landmarks_1[m], fmt='%d')
                np.savetxt(os.path.join(current_landmarks_save_dir,txt_name2), current_landmarks_2[m], fmt='%d')
                np.savetxt(os.path.join(current_landmarks_save_dir,txt_name3), current_landmarks_3[m], fmt='%d') 
                np.savetxt(os.path.join(current_landmarks_save_dir,txt_name4), current_landmarks_4[m], fmt='%d')
                # print(current_landmarks.shape)  (2, 68, 2)

                in_land = torch.mm(self.real_landmarks, torch.t(self.PCA)) + self.mean
                # print(in_land.shape)    # (1, 162)
                in_land = in_land.detach().cpu().numpy() * 256
                in_land = in_land[m].reshape((81, 2))

              
      
                img_landmarks_name = img_name + '_trans_to_class_' + str(i) + '_Land' + '.png'
                img_landmarks = util.landmarks2im(in_land, current_landmarks[m], current_landmarks_1[m], current_landmarks_2[m], current_landmarks_3[m], current_landmarks_4[m])
          
                cv2.imwrite(os.path.join(img_landmarks_save_dir, img_landmarks_name), img_landmarks)
               






    def get_visuals(self):
        return_dicts = [OrderedDict() for i in range(self.numValid)]
    
        real_A = util.tensor2im(self.reals.data)
       
        # 1
        fake_B_tex = util.tensor2im(self.fake_B.data)
        fake_B_tex1 = util.tensor2im(self.fake_B1.data)
        fake_B_tex2 = util.tensor2im(self.fake_B2.data)
        fake_B_tex3 = util.tensor2im(self.fake_B3.data)
        

        if self.debug_mode:   # --debug_mode', action='store_true
            rec_A_tex = util.tensor2im(self.cyc_A.data[:,:,:,:,:])

        if self.numValid == 1:
            real_A = np.expand_dims(real_A, axis=0)

        # for i in range(self.numValid):   
        # for i in range(2):
        # get the original image and the results for the current samples
        curr_real_A = real_A[:, :, :]
        real_A_img = curr_real_A[:, :, :3]

        # start with age progression/regression images
        if self.traverse or self.deploy:
            curr_fake_B_tex = fake_B_tex
            orig_dict = OrderedDict([('orig_img', real_A_img)])
        else:
            
            # 1
            curr_fake_B_tex = fake_B_tex[:, 0, :, :, :]
            curr_fake_B_tex1 = fake_B_tex1[:, 0, :, :, :]
            curr_fake_B_tex2 = fake_B_tex2[:, 0, :, :, :]
            curr_fake_B_tex3 = fake_B_tex3[:, 0, :, :, :]
           

            orig_dict = OrderedDict([('orig_img_cls_' + str(self.class_A[0].item()), real_A_img)])

        return_dicts[0].update(orig_dict)

        # set output classes numebr
        if self.traverse:
            out_classes = curr_fake_B_tex.shape[0]
        else:
            out_classes = self.numClasses

        for j in range(out_classes):   
            
            # 1
            fake_res_tex = curr_fake_B_tex[j, :, :, :3]
            fake_res_tex1 = curr_fake_B_tex1[j, :, :, :3]
            fake_res_tex2 = curr_fake_B_tex2[j, :, :, :3]
            fake_res_tex3 = curr_fake_B_tex3[j, :, :, :3]
            
            #1
    
            fake_dict_tex = OrderedDict([('tex_trans_to_class_' + str(j)+'_0', fake_res_tex)])
            return_dicts[0].update(fake_dict_tex)

            fake_dict_tex1 = OrderedDict([('tex_trans_to_class_' + str(j)+'_1', fake_res_tex1)])
            return_dicts[0].update(fake_dict_tex1)
           
            fake_dict_tex2 = OrderedDict([('tex_trans_to_class_' + str(j)+'_2', fake_res_tex2)])
            return_dicts[0].update(fake_dict_tex2)

            fake_dict_tex3 = OrderedDict([('tex_trans_to_class_' + str(j)+'_3', fake_res_tex3)])
            return_dicts[0].update(fake_dict_tex3)


        if not (self.traverse or self.deploy):
            if self.debug_mode:
                # continue with tex reconstructions
                curr_rec_A_tex = rec_A_tex[:, 0, :, :, :]
                orig_dict = OrderedDict([('orig_img2', real_A_img)])
                return_dicts[0].update(orig_dict)
                for j in range(self.numClasses):
                    rec_res_tex = curr_rec_A_tex[j, :, :, :3]
                    rec_dict_tex = OrderedDict([('tex_rec_from_class_' + str(j), rec_res_tex)])
                    return_dicts[0].update(rec_dict_tex)

        return return_dicts


class InferenceModel(LATS):
    def forward(self, data):
        return self.inference(data)
