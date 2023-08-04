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
    def __init__(self, dataset_jsonfile_path):
        self.datapath = dataset_jsonfile_path
        with open(dataset_jsonfile_path, 'r') as f:
            data_json = json.load(f)
        
        self.data = data_json['data']
        self.data = self.dict_to_numpy_data(self.data)
        pdb.set_trace()
    
    def __len__(self):
        return
    
    def __repr__(self):
        return "Event Dataset"

    def __getitem__(self, index):
        # use a single frame with one wheel accel segment, for now
        datum = self.data[index]
        datum = self.unpack_data[datum]
        wheel_accel_spec = self.wheel_accel2spec(datum['wheel_accel_path'])
        
        return
    
    def preprocess(self):
        return 
    
    def wheel_accel2spec(self, wheel_accel_path):
        # load np array from wheel_accel_path
        sampling_freq = 500
        N_window_FFT = 32
        plt.figure()
        spectrum, freqs, t_bins, im = plt.specgram(x=wheel_accel, 
                                                    NFFT=N_window_FFT, 
                                                    noverlap=0, 
                                                    Fs=sampling_freq, 
                                                    Fc=0,
                                                    mode='default',
                                                    scale='default',
                                                    scale_by_freq=True) # (17,16) or (33,8)
        return spectrum
    
    def dict_to_numpy_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['frame_id'], data_json[i]['frame_path'], data_json[i]['wheel_accel_path'], data_json[i]['event_label']]
        data_np = np.array(data_json, dtype=str)
        return data_np
    
    def unpack_data(self, data_np):
        datum = {}
        datum['frame_id'], datum['frame_path'], datum['wheel_accel_path'], datum['event_label'] = data_np
        return datum


