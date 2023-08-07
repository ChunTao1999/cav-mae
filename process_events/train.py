# Author: Chun Tao

#%% Imports
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
from dataloader_events import RoadEventDataset
from preprocess_data_folder import preprocess
from utils import calibrate_camera, define_perspective_transform
import pdb # debug

#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--calibrate', type=int, default=0, required=True, help='whether to calibrate camera and save camera matrices, 0 for True')
parser.add_argument('-p', '--perspective', type=int, default=0, required=True, help='whether to compute perspective transform or not, 0 for True')
parser.add_argument('-j', '--dataset-jsonfile-path', type=str, default='', required=True, help='filepath to dataset jsonfile')
parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
parser.add_argument('--cal-data-path', type=str, default='', required=True, help='path to the saved calibration data')
parser.add_argument('--dataset-path', type=str, default='', required=True, help='dataset path to save to')
parser.add_argument('--download-csvs', type=int, default=0, required=True, help='whether to download session csvs for sensor data, or to use a past stored session csv, 0 for True')
parser.add_argument('--session-id', type=int, default=75151, required=False, help='session ids to the session data')
parser.add_argument('--wheelaccel-timespan', type=float, default=1.0, required=True, help='the timespan of WheelAccel segments for each event')
args = parser.parse_args()


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
    print("Preprocess: perspective transform......")
    define_perspective_transform(cal_data_path=args.cal_data_path,
                                img_cal_path=os.path.join(args.data_path, 'cal_img.jpg'),
                                IMAGE_H=600, 
                                IMAGE_W=600,
                                srcpts_arr = [[236,1218],[2170,1190],[1532,703],[885,708]], # CCLK, from bottom left
                                destpts_arr= [[50,580],[550,580],[550,250],[50,250]])
else:
    print("Preprocess: perspective transform skipped, using previous......")
# define wheelAccel transform configs
wheelAccel_conf = {'wheel_id': ['rlWheelAccel', 'rrWheelAccel'],
                   'sampling_freq': 500,
                   'timespan': args.wheelaccel_timespan,
                   'N_windows_fft': 32,
                   'noverlap': 16,
                   'spec_size': (17, 31)}
# Preprocess to get event frames, event 1-d wheelAccel, event 2-d spectrograms, and grouth-truth bbox locations and event labels
# Write all info to the dataset dictionary
preprocess(cal_data_path=args.cal_data_path,
           data_path=args.data_path,
           save_path=args.dataset_path,
           wheelAccel_conf=wheelAccel_conf,
           download=True if args.download_csvs==0 else False,
           plot_wheelAccel=True)

roadevent_dataset = RoadEventDataset(args.dataset_jsonfile_path)
