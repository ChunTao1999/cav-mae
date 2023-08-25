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
import torchaudio
import pdb # debug


#%% Event dataset class
class RoadEventDataset(Dataset):
    """Dataset that manages sensor data segments and corresponding event frames"""
    def __init__(self, dataset_jsonfile_path, dataset_conf):
        self.datapath = dataset_jsonfile_path
        with open(dataset_jsonfile_path, 'r') as f:
            data_json = json.load(f)
        # preprocess the weel accel data to get spectrograms
        self.data = data_json['data']
        self.event_ids = list(self.data.keys())   
        self.frame_list = self.get_frame_list()
        self.frame_size = dataset_conf['frame_size']
        self.frame_transform = T.Compose([T.Resize(size=self.frame_size,
                                                   interpolation=T.InterpolationMode.BILINEAR),
                                          T.ToTensor()])
        self.spec_toTensor = T.ToTensor()


    def get_frame_list(self):
        frame_list = []
        for event_id in self.event_ids:
            for frame_path in self.data[event_id]['frame_paths']:
                frame_list.append(frame_path)    
        return frame_list    


    def load_spec(self, wheel_accel_path):
        """Load wheel accel 1-d data and its corresponding 2-d spectrogram"""
        spec = np.load(wheel_accel_path)
        return spec
    

    def __len__(self):
        return len(self.frame_list)
    

    def __repr__(self):
        return "Event Dataset"


    def bbox_transform(self, bbox, orig_size, new_size):
        bbox = bbox[:, [1,0]]
        bbox[:,0] *= (new_size[0]/orig_size[0]) # height
        bbox[:,1] *= (new_size[1]/orig_size[1]) # width
        return bbox


    def __getitem__(self, index):
        frame_path = self.frame_list[index]
        frame_idx = int(frame_path.split('/')[-1].split('_')[5])
        event_id = '_'.join(frame_path.split('/')[-1].split('_')[:4])
        datum = self.data[event_id] # datum key is the event_id
        wheelAccel_spec = self.load_spec(datum['wheelAccel_spec_path'])
        wheelAccel_spec = self.spec_toTensor(wheelAccel_spec)

        event_frame = Image.open(frame_path) 
        event_frame = self.frame_transform(event_frame) 

        # get the labels from dict'
        event_label = np.array(int(datum['event_label'])) # string of "0" to "4"
        event_label_tensor = torch.zeros(1, dtype=torch.float)
        event_label_tensor = torch.from_numpy(event_label)
        bbox = np.array(datum['bbox_coords'][frame_idx])
        bbox_tensor = torch.from_numpy(bbox)
        bbox_tensor = self.bbox_transform(bbox_tensor, orig_size=(1080, 1920), new_size=self.frame_size) # (height, width)
        bbox_tensor = bbox_tensor.flatten()
        
        # return the event frame, event spec, event bbox coords, and event label
        return wheelAccel_spec, event_frame, event_label_tensor, bbox_tensor