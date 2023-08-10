# Author: Chun Tao

#%% Imports
import argparse
import json
import os
import random
from PIL import Image
import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from dataloader_events import RoadEventDataset
from preprocess_data import preprocess
from utils import calibrate_camera, define_perspective_transform
import pdb # debug

#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--calibrate', type=int, default=0, required=True, help='whether to calibrate camera and save camera matrices, 0 for True')
parser.add_argument('-p', '--perspective', type=int, default=0, required=True, help='whether to compute perspective transform or not, 0 for True')
parser.add_argument('-j', '--dataset-jsonfile-path', type=str, default='', required=True, help='filepath to dataset jsonfile')
parser.add_argument('-s', '--seed', type=int, default=0, required=True, help='the seed number to use for the current runtime')
parser.add_argument('-e', '--num-epochs', type=int, default=1, required=True, help='number of epochs for training')
parser.add_argument('--preprocess', type=int, default=0, required=True, help='whether or not to preprocess wheelAccels and event frames, 0 for True')
parser.add_argument('--eventtype-json-path', type=str, default='', required=True, help='path to the json file describing convertion from event type label to event type description')
parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
parser.add_argument('--cal-data-path', type=str, default='', required=True, help='path to the saved calibration data')
parser.add_argument('--dataset-path', type=str, default='', required=True, help='dataset path to save to')
parser.add_argument('--download-csvs', type=int, default=0, required=True, help='whether to download session csvs for sensor data, or to use a past stored session csv, 0 for True')
parser.add_argument('--session-id', type=int, default=75151, required=False, help='session ids to the session data')
parser.add_argument('--wheelaccel-timespan', type=float, default=1.0, required=True, help='the timespan of WheelAccel segments for each event')
args = parser.parse_args()
#%% Seeds
SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benmarks=False
os.environ['PYTHONHASHSEED']=str(SEED)
#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Preprocesses
# if needed, calibrate the camera and save camera matrices
if not os.path.exists(os.path.join(args.data_path, "results")):
    os.makedirs(os.path.join(args.data_path, "results"))
if args.calibrate==0:
    print("Preprocess: camera calibration......")
    calibrate_camera(objp_w=10,
                     objp_h=7,
                     cal_data_path=args.cal_data_path,
                     save_path=os.path.join(args.data_path, "results"))
else:
    print("Preprocess: camera calibration skipped, using previous......")
# define the perspective transform to use for the raw rv road event frames
if args.perspective==0:
    print("\nPreprocess: perspective transform......")
    define_perspective_transform(cal_data_path=args.cal_data_path,
                                 img_cal_path=os.path.join(args.data_path, 'cal_img.jpg'),
                                 IMAGE_H=600, 
                                 IMAGE_W=600,
                                 cv2_imread_frame_dim=(1920, 1080),
                                 srcpts_arr = [[236,1218],[2170,1190],[1532,703],[885,708]], # CCLK, from bottom left
                                 destpts_arr= [[50,580],[550,580],[550,250],[50,250]])
else:
    print("\nPreprocess: perspective transform skipped, using previous......")
# define wheelAccel transform configs
wheelAccel_conf = {'wheel_id': ['rlWheelAccel', 'rrWheelAccel'],
                   'sampling_freq': 500,
                   'timespan': args.wheelaccel_timespan,
                   'normalize': True,
                   'nominal_speed': 5,
                   'N_windows_fft': 32,
                   'noverlap': 16,
                   'spec_size': (17, 31)}
eventMarking_conf = {'cv2_imread_frame_dim': (1920, 1080), # (w, h)
                     'event_timestamp_shift': 0, # negative shift here
                     'frame_timestamp_shift': 0.2,
                     'bev_frame_dim': (600,600),
                     'wheel_to_base_dist': 4, # 4.572
                     'base_pixel': 20,
                     'wheel_width': 1.664,
                     'xm_per_pix': 4.318/500,
                     'ym_per_pix': 8.8/330, # 8.89
                     'event_len_pix': 200}

# Preprocess to get event frames, event 1-d wheelAccel, event 2-d spectrograms, and grouth-truth bbox locations and event labels
# Write all info to the dataset dictionary
if args.preprocess==0:
    print("\nPreprocess: transforming wheelAccel to spectrogram and marking events in the frames......")
    preprocess(cal_data_path=args.cal_data_path,
               data_path=args.data_path,
               save_path=args.dataset_path,
               json_save_path = args.dataset_jsonfile_path,
               wheelAccel_conf=wheelAccel_conf,
               eventmarking_conf=eventMarking_conf,
               eventType_json_path=args.eventtype_json_path,
               download=True if args.download_csvs==0 else False,
               plot_wheelAccel=True,
               plot_processedFrames=True)
else:
    print("\nPreprocess: transforming wheelAccel to spectrogram and marking events in the frames skipped, using existing dataset......")


#%% RoadEvent Dataset
# Initiate RoadEvent Dataset
dataset_conf = {'frame_size': (256, 256),
                'batch_size': 4,
                'shuffle': True,
                'num_workers': 4} # frame_size
roadevent_dataset_train = RoadEventDataset(dataset_jsonfile_path=args.dataset_jsonfile_path,
                                           dataset_conf=dataset_conf) # pdb breakpoint inside
print(f'\nTrain dataset created, size: {roadevent_dataset_train.__len__()}')
sample_spec, sample_frame = roadevent_dataset_train.__getitem__(0)
print(f'\twheelAccel spec size: {sample_spec.shape}')
print(f'\tframe size: {sample_frame.shape}')

# Initiate RoadEvent DataLoader
roadevent_dataloader_train = DataLoader(dataset=roadevent_dataset_train,
                                        batch_size=dataset_conf['batch_size'],
                                        shuffle=dataset_conf['shuffle'],
                                        num_workers=dataset_conf['num_workers'])

# Training and validation loop
for epoch in range(args.num_epochs):
    regression_loss = 0
    classfication_loss = 0
    for sample_idx, data in enumerate(roadevent_dataloader_train):
        spec, img = data[0].float().to(device), data[1].to(device)
        
        pdb.set_trace()



# TO-DOs:
# use tkinter to develop a GUI that allows the user to manually tweak the bounding box annotation and road event classification label
# IOU loss on bbox reconstruction and CE loss on event classification, produce MaP and Confusion Matrix
pdb.set_trace()
