import cv2
import json
import numpy as np
import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
import shutil
import pdb # for debug

# load a previous session json metafile
prev_meta_path = "/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_9.5.json"
new_meta_path = "/home/nano01/a/tao88/RoadEvent-Dataset-Public/9.5"
annotated_path = "/home/nano01/a/tao88/RoadEvent-shared/CV/"
sub_folders_list = ["datafiles", "undistorted_rv", "undistorted_rv_annotated", "undistorted_bev_annotated"]
for sub_folder_name in sub_folders_list:
    if not os.path.exists(os.path.join(new_meta_path, sub_folder_name)): os.makedirs(os.path.join(new_meta_path, sub_folder_name))
new_dict = {}
session_id_list = [75151, 75208, 75211, 75212, 75213, 75214, 75216, 75218, 75219, 75223, 75224, 75225, 75226, 75307, 75308, 75309, 75310, 75316, 75317, 75318, 75319, 75366, 75367, 75413, 75414, 75415]
date_list = ["7.26", "8.3", "8.3", "8.3", "8.4", "8.4", "8.4", "8.6", "8.6", "8.6", "8.6", "8.6", "8.6", "8.12", "8.12", "8.12", "8.12", "8.12", "8.12", "8.12", "8.12", "8.18", "8.18", "8.31", "8.31", "8.31"]
# session_id_list = [75413, 75414, 75415]
# date_list = ["8.31", "8.31", "8.31"]
prev_meta = json.load(open(prev_meta_path))['data']

# load csv containing session timeshift
session_shift_dict = {}
date_dict = {}
for idx, session_id in enumerate(session_id_list):    
    json_path = os.path.join("/home/nano01/a/tao88/RoadEvent-shared/CV/session_csvs/", "unpackedSession_{}.json".format(session_id))
    session_timeShift = json.load(open(json_path, 'r'))[0]['timeOffsetShift']
    session_timeShift = float(format(session_timeShift, '.3f'))
    session_shift_dict[session_id] = session_timeShift
    date_dict[session_id] = date_list[idx]


processed_sample_count = 0
# output updated metafile
for key, value in prev_meta.items():
    processed_sample_count += 1
    sessionId = key.split('_')[1]
    session_timeShift = session_shift_dict[int(sessionId)]
    eventTime = float(key.split('_')[3]) - session_timeShift
    new_key = f"s{sessionId}_{eventTime:.3f}".replace(".", "")
    new_dict[new_key] = {}
    new_dict[new_key]['frames'] = []
    date = date_dict[int(sessionId)]
    for frame_str in value['frame_paths']:
        frame_name = frame_str.split('/')[-1]
        frame_rv_annotated_name = '.'.join(frame_name.split('.')[:-1])+'_rv.png'
        frame_bev_annotated_name = '.'.join(frame_name.split('.')[:-1])+'_bev.png'
        frame_idx, frame_time, frame_dist = frame_name[:-4].split('_')[5], \
                                            frame_name[:-4].split('_')[8], \
                                            frame_name[:-4].split('_')[10]
        new_frame_name = f"{new_key}_{frame_idx}_{(float(frame_time) - session_timeShift):.3f}_{float(frame_dist):.3f}".replace(".", "")

        # update new dict frame name
        new_dict[new_key]['frames'].append(new_frame_name)
        # copy undistorted rv frames to the public dataset folder
        shutil.copy(frame_str, os.path.join(new_meta_path, "undistorted_rv", new_frame_name+'.png'))
        # copy undistorted rv frames annotated to the public dataset folder
        shutil.copy(os.path.join(annotated_path, "events_"+date, "results", "frames_rv_annotated", frame_rv_annotated_name), \
                    os.path.join(new_meta_path, "undistorted_rv_annotated", new_frame_name+'.png'))
        # copy undistorted bev frames annotated to the public dataset folder
        shutil.copy(os.path.join(annotated_path, "events_"+date, "results", "frames_bev_annotated", frame_bev_annotated_name), \
                    os.path.join(new_meta_path, "undistorted_bev_annotated", new_frame_name+'.png'))
    if processed_sample_count % 100 == 0:
        print(f"{processed_sample_count} / {len(prev_meta.keys())}")

print("Processed count: " + str(processed_sample_count))

# save updated dict
with open(os.path.join(new_meta_path, "datafiles", "events_metafile.json"), 'w') as outfile:
    json.dump(new_dict, outfile)