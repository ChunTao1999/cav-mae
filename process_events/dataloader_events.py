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
import torch.nn.functional
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
        self.frame_size = dataset_conf['frame_size']
        self.frame_transform = T.Compose([T.Resize(size=self.frame_size,
                                                   interpolation=T.InterpolationMode.BILINEAR),
                                          T.ToTensor()])
        self.spec_transform = T.Compose([T.ToTensor()])


    def preprocess(self):
        return 
    

    def load_spec(self, wheel_accel_path):
        """Load wheel accel 1-d data and its corresponding 2-d spectrogram"""
        spec = np.load(wheel_accel_path)
        return spec
    
    def __len__(self):
        return len(self.data)
    

    def __repr__(self):
        return "Event Dataset"


    def __getitem__(self, index):
        # use a single frame with one wheel accel segment, for now
        datum = self.data[index] # datum key is the event_id
        event_id = list(datum)[0]
        wheelAccel_spec = self.load_spec(datum[event_id]['wheelAccel_spec_path'])
        wheelAccel_spec = self.spec_transform(wheelAccel_spec)
        event_frame = Image.open(datum[event_id]['frame_paths'][0])
        event_frame = self.frame_transform(event_frame) # use the first frame (nearest)
    
        
        # return the event frame, event spec, event bbox coords, and event label
        return wheelAccel_spec, event_frame