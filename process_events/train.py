# Author: Chun Tao

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
import pdb # debug


parser = argparse.ArgumentParser()
parser.add_argument('-j', '--dataset-jsonfile-path', type=str, default='', required=True, help='filepath to dataset jsonfile')
parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
parser.add_argument('--cal-data-path', type=str, default='', required=True, help='path to the saved calibration data')
parser.add_argument('--download-csvs', type=int, default=0, required=True, help='whether to download session csvs for sensor data, or to use a past stored session csv, 0 for True')
parser.add_argument('--session-id', type=int, default=75151, required=False, help='session ids to the session data')
parser.add_argument('--wheelaccel-timespan', type=float, default=1.0, required=True, help='the timespan of WheelAccel segments for each event')
args = parser.parse_args()

# access the data folder and get the session ids
wheelAccel_conf = {'wheel_id': ['rlWheelAccel', 'rrWheelAccel'],
                   'sampling_freq': 500,
                   'timespan': args.wheelaccel_timespan,
                   'N_windows_fft': 32,
                   'spec_size': (64, 64)}
preprocess(args.data_path,
           wheelAccel_conf=wheelAccel_conf,
           download=True if args.download_csvs==0 else False,
           plot_wheelAccel=True)

roadevent_dataset = RoadEventDataset(args.dataset_jsonfile_path)

# add to dataset class
#         # np.fft or plt.specgram both produces spectrogram
#         sampling_freq = 500
#         N_window_FFT = 32
#         plt.figure()
#         spectrum, freqs, t_bins, im = plt.specgram(x=wheel_accel, 
#                                                     NFFT=N_window_FFT, 
#                                                     noverlap=0, 
#                                                     Fs=sampling_freq, 
#                                                     Fc=0,
#                                                     mode='default',
#                                                     scale='default',
#                                                     scale_by_freq=True) # (17,16) or (33,8)
#         plt.savefig('/home/nano01/a/tao88/cav-mae/process_events/figures/wheel_accel/wheel_accel_event_{:.3f}_spec.png'.format(event_timestamp))
