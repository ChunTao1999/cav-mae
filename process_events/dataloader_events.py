# -*- coding: utf-8 -*-
# @Time    :      7/26/23 1:26 PM
# @Author  :      Chun Tao
# @Affiliation  : Purdue University
# @Email   :      tao88@purdue.edu
# @File    :      dataloader_events.py

# modified from:
# Author: Yuan Gong

import argparse
import csv
import glob
import json
import os
import random
import PIL
from PIL import Image
import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnF
from torch.utils.data import Dataset
import torchvision.transforms as T
# import torchaudio
import pdb # debug


# Up-to-Date
class RoadEventDataset(Dataset):
    """Dataset that manages sensor data segments and corresponding event frames"""
    def __init__(self, dataset_metafile_path, data_folder_path, split_dict, dataset_conf, split="train"):
        assert split in ["train", "val", "test"], "Dataset split type not valid!"
        self.split = split
        self.split_dict = split_dict
        self.metafilePath = dataset_metafile_path
        with open(self.metafilePath, 'r') as f:
            self.data = json.load(f)

        # taxonomy
        # {'label': '4', 'type': 'negative event', 'wheel': '3', 'speed': 5.019, 'settling_time': 0.11, 'settling_dist': 0.6024444444444444, 'p2p_time': 0.034, 'frames': ['s_75151_24473_0_25488_3', 's_75151_24473_1_25984_3', 's_75151_24473_2_26551_3'], 'bev_boxcenter': [[402.30811469431126, 457.2730729879391], [405.6064696428514, 355.13187405636796], [410.17175695700075, 236.73277720433828]], 'bev_rot_rect_box': [[[334.064453125, 405.4857482910156], [475.3176574707031, 406.1719665527344], [329.2985534667969, 508.3741760253906], [470.5517578125, 509.0603942871094]], [[337.9245300292969, 303.2673034667969], [479.16943359375, 304.1141052246094], [332.04351806640625, 406.149658203125], [473.2884216308594, 406.9964599609375]], [[343.06036376953125, 184.79049682617188], [484.2950744628906, 185.80010986328125], [336.0484313964844, 287.6654357910156], [477.28314208984375, 288.675048828125]]], 'rv_poly_box': [[[1058.2322998046875, 573.84765625], [1278.532470703125, 572.6364135742188], [1072.1126708984375, 684.3544921875], [1387.5213623046875, 682.3692016601562]], [[1049.779052734375, 514.920654296875], [1219.3690185546875, 514.1314697265625], [1055.181396484375, 574.3707275390625], [1276.0438232421875, 573.2763671875]], [[1044.460693359375, 473.4711608886719], [1178.3231201171875, 472.9212646484375], [1045.9600830078125, 508.25738525390625], [1209.8560791015625, 507.5675964355469]]], 'rv_rot_rect_box': [[[1387.5213623046873, 682.3692016601562], [1386.8252720171843, 571.779379335446], [1058.2322998046875, 573.8476562499999], [1058.9283900921905, 684.4374785747101]], [[1276.0438232421875, 573.2763671874999], [1275.7491262211552, 513.8009858557834], [1049.779052734375, 514.920654296875], [1050.0737497554073, 574.3960356285914]], [[1209.8560791015625, 507.5675964355469], [1209.7096504091553, 472.7756778656865], [1044.460693359375, 473.4711608886719], [1044.6071220517822, 508.2630794585323]]], 'rv_rot_rect_box_dim': [[1222.8768310546873, 628.1084289550781, 1.5645020667956901, 110.59201301856888, 328.5994813703119, 36340.478121600514], [1162.9114379882812, 544.0985107421874, 1.565841426082642, 59.47611143044992, 225.97284741543353, 13439.986253136381], [1127.1583862304688, 490.5193786621094, 1.5665876532332053, 34.792226705598466, 165.2504205824464, 5749.43009609997]], 'difficult': [0, 0, 0]}
        # 'rv_rot_rect_box_dim' saved as (x_c, y_c, yaw, w, h, a)

        self.dataFolderPath = data_folder_path
        self.event_ids = list(self.data.keys())   
        self.frame_list, self.neg_count, self.pos_count = self.get_frame_list() # get the frames correponding to self.split, neg_count=643, pos_count=132
        # self.loss_ratio = [len(self.event_ids)/self.neg_count, len(self.event_ids)/self.pos_count]
        self.frame_type = dataset_conf['frame_type']
        assert (self.frame_type in ["undistorted_rv", 
                                    "undistorted_rv_annotated",
                                    "undistorted_bev_annotated"]), "frame_type unknown!"
        self.frame_size = dataset_conf['frame_size']
        self.frame_transform = T.Compose([T.Resize(size=self.frame_size,
                                                   interpolation=T.InterpolationMode.BILINEAR),
                                          T.ToTensor(),
                                          T.Normalize(mean=dataset_conf['normalization']['mean'],
                                                      std=dataset_conf['normalization']['std'])])
        self.spec_toTensor = T.ToTensor()


    def get_frame_list(self):
        frame_list = []
        neg_count, pos_count = 0, 0
        for event_id in self.event_ids:
            if self.data[event_id]['label'] == '4':
                neg_count += 1
            elif self.data[event_id]['label'] == '5':
                pos_count += 1
            for frame_path in self.data[event_id]['frames']:
                if self.split == "train" and self.split_dict[frame_path] == 0:
                    frame_list.append(frame_path)
                elif self.split == "val" and self.split_dict[frame_path] == 1:
                    frame_list.append(frame_path)
                elif self.split == "test" and self.split_dict[frame_path] == 2:
                    frame_list.append(frame_path)
        return frame_list, neg_count, pos_count


    def load_spec(self, wheel_accel_path):
        """Load wheel accel 1-d data and its corresponding 2-d spectrogram"""
        spec = np.load(wheel_accel_path)
        return spec
    

    def __len__(self):
        return len(self.frame_list)
    

    def __repr__(self):
        return "Event Dataset"


    def bbox_transform(self, bbox, orig_size, new_size):
        bbox[0] *= (new_size[0]/orig_size[0]) # x_c
        bbox[1] *= (new_size[1]/orig_size[1]) # y_c
        bbox[3] *= (new_size[0]/orig_size[0]) # w
        bbox[4] *= (new_size[1]/orig_size[1]) # h
        return bbox


    def __getitem__(self, index):
        # get frame and event ids from frame name
        frame_name = self.frame_list[index]
        frame_idx = int(frame_name.split('_')[3])
        event_id = '_'.join(frame_name.split('_')[:3])
        data = self.data[event_id]

        # load event frame and event motion characteristics, perform necessary transforms (11.20)
        event_frame = Image.open(os.path.join(self.dataFolderPath, self.frame_type, frame_name+'.png'))
        event_frame = self.frame_transform(event_frame) 

        # prepare wheelAccel 1-D profile if necessary

        # can deal with difficult labels too

        # prepare labels:
        # 1) bounding box dimensions (x_c, y_c, alpha, w, h) from 'rv_rot_rect_box_dim'
        # 2) event type label from 'label' or 'type', 4 for negative event, 5 for positive event
        # 3) motion characteristics: 
        #   a) speed at event instant from 'speed'
        #   b) settling time of the wheelAccel signal from 'settling_time'
        #   c) setting distance of the wheelAccel signal from 'settling_dist'
        #   d) peak-to-peak time in the wheelAccel signal window, starting from the event instant, from 'p2p_time'
        # Note (11.25): need to transform the bbox dimensions
        rv_rotated_bbox = torch.tensor(data['rv_rot_rect_box_dim'][frame_idx])[:-1].float() # (5,), (x_c, y_c, alpha, w, h)
        rv_rotated_bbox = self.bbox_transform(rv_rotated_bbox, orig_size=(1920,1080), new_size=self.frame_size)
        # event_cls_lbl = torch.tensor(int(data['label'])-4).type(torch.LongTensor) # (1,)
        event_cls_lbl = torch.tensor(int(data['label'])-4).float() # 0 for negative, 1 for positive
        event_motion_lbl = torch.tensor([data['speed'], data['settling_time'], data['settling_dist'], data['p2p_time']]).float() # (4,), (speed_at_event, settling time, settling distance, p2p time)

        # return transformed frame, transformed motion, and GT labels
        return event_frame, [rv_rotated_bbox, event_cls_lbl, event_motion_lbl]
        

# Original
# class RoadEventDataset(Dataset):
#     """Dataset that manages sensor data segments and corresponding event frames"""
#     def __init__(self, dataset_jsonfile_path, dataset_conf):
#         self.datapath = dataset_jsonfile_path
#         with open(dataset_jsonfile_path, 'r') as f:
#             data_json = json.load(f)
#         # preprocess the weel accel data to get spectrograms
#         self.data = data_json['data']
#         self.event_ids = list(self.data.keys())   
#         self.frame_list = self.get_frame_list()
#         self.frame_size = dataset_conf['frame_size']
#         self.frame_transform = T.Compose([T.Resize(size=self.frame_size,
#                                                    interpolation=T.InterpolationMode.BILINEAR),
#                                           T.ToTensor()])
#         self.spec_toTensor = T.ToTensor()


#     def get_frame_list(self):
#         frame_list = []
#         for event_id in self.event_ids:
#             for frame_path in self.data[event_id]['frame_paths']:
#                 frame_list.append(frame_path)    
#         return frame_list    


#     def load_spec(self, wheel_accel_path):
#         """Load wheel accel 1-d data and its corresponding 2-d spectrogram"""
#         spec = np.load(wheel_accel_path)
#         return spec
    

#     def __len__(self):
#         return len(self.frame_list)
    

#     def __repr__(self):
#         return "Event Dataset"


#     def bbox_transform(self, bbox, orig_size, new_size):
#         bbox = bbox[:, [1,0]]
#         bbox[:,0] *= (new_size[0]/orig_size[0]) # height
#         bbox[:,1] *= (new_size[1]/orig_size[1]) # width
#         return bbox


#     def __getitem__(self, index):
#         frame_path = self.frame_list[index]
#         frame_idx = int(frame_path.split('/')[-1].split('_')[5])
#         event_id = '_'.join(frame_path.split('/')[-1].split('_')[:4])
#         datum = self.data[event_id] # datum key is the event_id
#         wheelAccel_spec = self.load_spec(datum['wheelAccel_spec_path'])
#         wheelAccel_spec = self.spec_toTensor(wheelAccel_spec)

#         event_frame = Image.open(frame_path) 
#         event_frame = self.frame_transform(event_frame) 

#         # get the labels from dict'
#         event_label = np.array(int(datum['event_label'])) # string of "0" to "4"
#         event_label_tensor = torch.zeros(1, dtype=torch.float)
#         event_label_tensor = torch.from_numpy(event_label)
#         bbox = np.array(datum['bbox_coords'][frame_idx])
#         bbox_tensor = torch.from_numpy(bbox)
#         bbox_tensor = self.bbox_transform(bbox_tensor, orig_size=(1080, 1920), new_size=self.frame_size) # (height, width)
#         bbox_tensor = bbox_tensor.flatten()
        
#         # return the event frame, event spec, event bbox coords, and event label
#         return wheelAccel_spec, event_frame, event_label_tensor, bbox_tensor