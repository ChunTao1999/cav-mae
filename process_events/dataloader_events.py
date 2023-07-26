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
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio
import pdb # debug

#%% Arguments
parser = argparse.ArgumentParser(description='function to create a PyTorch dataset for road event/anomaly detection and classification')
parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
parser.add_argument('--cal-data-path', type=str, default='', required=True, help='path to the saved calibration data')
parser.add_argument('--download-csvs', type=int, default=0, required=True, help='whether to download session csvs for sensor data, or to use a past stored session csv, 0 for True')
parser.add_argument('--session-id', type=int, default=75151, required=False, help='session ids to the session data')
args = parser.parse_args()


#%% Event dataset class
class RoadEventDataset(Dataset):
    """Dataset that manages sensor data segments and corresponding event frames"""
    def __init__(self, dataset_json_path):
        self.datapath = dataset_json_path
    
    def __len__(self):
        return
    
    def __repr__(self):
        return "Event Dataset"

    def __getitem__(self, index):
        return


#%% Main program
if __name__ == '__main__':
    # Download the session csvs
    dataFolderNames = [a for a in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, a))]
    try:
        dataFolderNames.remove('results')
    except:
        pass
    try:
        dataFolderNames.remove('chessboards')
    except:
        pass
    try:
        dataFolderNames.remove('session_csvs')
    except:
        pass
    csv_save_path = os.path.join('/'.join(args.data_path.split('/')[:-1]), 'session_csvs')
    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)
    if args.download_csvs == 0:
        # download 100Hz session data, session offset data, and session events data
        sessionIds = ' '.join(s.split('_')[-1] for s in dataFolderNames)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh {}".format(sessionIds), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh {}".format(sessionIds), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh {}".format(sessionIds), shell=True)

    # get the rear wheel accel data from session data, and FFT to get spectrogram (freq vs. time)
    




    pdb.set_trace()
    


