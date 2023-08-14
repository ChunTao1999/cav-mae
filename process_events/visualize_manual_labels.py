import argparse
import json
import cv2
import numpy as np
import os
from utils import add_bbox_to_frame
import pdb # for debug

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--calibration_path', type=str, default='', required=True, help='path to the calibration params')
parser.add_argument('-j', '--meta-json_file', type=str, default='', required=True, help='path to the manually labeled metafile')
parser.add_argument('-s', '--frame-save-path', type=str, default='', required=True, help='path to save the manually labeled event frames')
args = parser.parse_args()
if not os.path.exists(args.frame_save_path):
    os.makedirs(args.frame_save_path)

cal_npz = np.load(os.path.join(args.calibration_path, 'caldata.npz'))
mtx, dist = cal_npz['x'], cal_npz['y']

with open(args.meta_json_file, 'r') as in_file:
    meta_dict = json.load(in_file)

eventId_list = list(meta_dict['data'].keys())
for eventId in eventId_list:
    event_label = meta_dict['data'][eventId]['event_label']
    event_type = meta_dict['data'][eventId]['event_type']
    for frame_idx, frame_path in enumerate(meta_dict['data'][eventId]['frame_paths']):
        frame_name = frame_path.split('/')[-1]
        frame = cv2.imread(frame_path)
        frame = cv2.undistort(frame, mtx, dist)
        pts_inv = np.float32(meta_dict['data'][eventId]['bbox_coords'][frame_idx])
        frame_rv = add_bbox_to_frame(image=frame,
                                     pts_inv=pts_inv)
        frame_rv = cv2.putText(frame_rv, f"Event label: {event_label}, Event type: {event_type}",
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.frame_save_path, frame_name), frame_rv)
        print('\t{}'.format(frame_name))
print("All manually labeled rv images saved!")

