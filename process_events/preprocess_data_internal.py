# Author: Chun Tao
# Date: 8.1.2023

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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from dataloader_events import RoadEventDataset
from preprocess_data import preprocess, preprocess_internal
from utils import calibrate_camera, define_perspective_transform, plot_image, plot_conf_matrix, save_loss_tallies, plot_loss_curves
from utils import polygon_area, compute_intersection_area, compute_union_area
from models import EventNN, ModifiedResNet18
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pdb # debug


#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--calibrate', type=int, default=0, required=True, help='whether to calibrate camera and save camera matrices, 0 for True')
parser.add_argument('-p', '--perspective', type=int, default=0, required=True, help='whether to compute perspective transform or not, 0 for True')
parser.add_argument('-j', '--dataset-jsonfile-path', type=str, default='', required=True, help='filepath to dataset jsonfile')
parser.add_argument('-s', '--save-path', type=str, default='', required=True, help='filepath to save processed frames and datafiles')
parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
parser.add_argument('--preprocess', type=int, default=0, required=True, help='whether or not to preprocess wheelAccels and event frames, 0 for True')
parser.add_argument('--eventtype-json-path', type=str, default='', required=True, help='path to the json file describing convertion from event type label to event type description')
parser.add_argument('--cal-data-path', type=str, default='', required=True, help='path to the saved calibration data')
parser.add_argument('--dataset-path', type=str, default='', required=True, help='dataset path to save to')
parser.add_argument('--download-csvs', type=int, default=0, required=True, help='whether to download session csvs for sensor data, or to use a past stored session csv, 0 for True')
parser.add_argument('--wheelaccel-timespan', type=float, default=1.024, required=True, help='the timespan of WheelAccel segments for each event')
# training configs
parser.add_argument('--session-list', type=str, nargs='+', default=75151, required=True, help='list of session data to include in preprocess')
parser.add_argument('--date-list', type=str, nargs='+', default="7.26", required=True, help='list of dates, ordering corresponding to list of sessions')
args = parser.parse_args()


#%% Preprocesses
# If needed, calibrate the camera and save camera matrices
if not os.path.exists(os.path.join(args.dataset_path, "calibrate")):
    os.makedirs(os.path.join(args.dataset_path, "calibrate"))
if args.calibrate==0:
    print("Preprocess: camera calibration......")
    calibrate_camera(objp_w=10,
                     objp_h=7,
                     cal_data_path=args.cal_data_path,
                     save_path=os.path.join(args.dataset_path, "calibrate"))
else:
    print("Preprocess: camera calibration skipped, using previous......")
# define the perspective transform to use for the raw rv road event frames
if args.perspective==0:
    print("\nPreprocess: perspective transform......")
    define_perspective_transform(cal_data_path=args.cal_data_path,
                                 img_cal_path=os.path.join(args.dataset_path, "calibrate", 'cal_img.jpg'),
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
                     'frame_timestamp_shift': 0.3, #0.2
                     'bev_frame_dim': (600,600),
                     'wheel_to_base_dist': 3.5, # 4.572
                     'base_pixel': 20,
                     'track_width': 1.692,
                     'wheel_width':0.305, 
                     'wheel_diameter':0.686,
                     'xm_per_pix': 4.318/500,
                     'ym_per_pix': 8.8/330, # 8.89
                     'event_len_pix': 200,
                     'event_len_scale': 4}
# Preprocess to get event frames, event 1-d wheelAccel, event 2-d spectrograms, and grouth-truth bbox locations and event labels
# Write all info to the dataset dictionary
if args.preprocess==0:
    print("\nPreprocess: transforming wheelAccel to spectrogram and marking events in the frames......")
    preprocess_internal(session_list = args.session_list,
                        date_list = args.date_list,
                        cal_data_path=args.cal_data_path,
                        data_path=args.data_path,
                        save_path=args.save_path,
                        json_save_path = args.dataset_jsonfile_path,
                        wheelAccel_conf=wheelAccel_conf,
                        eventmarking_conf=eventMarking_conf,
                        eventType_json_path=args.eventtype_json_path,
                        download=True if args.download_csvs==0 else False,
                        plot_processedFrames=True)
else:
    print("\nPreprocess: transforming wheelAccel to spectrogram and marking events in the frames skipped, using existing dataset......")

