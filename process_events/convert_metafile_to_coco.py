# Author: Chun Tao
# Date: Mar 2024

#%% Imports
import argparse
import json
import os
import pdb

#%% Arguments
parser = argparse.ArgumentParser(description='Convert Metafile to COCO format')
parser.add_argument('--metafile_path', type=str, required=True, help='Path to the metafile')
args = parser.parse_args()

#%% Load metafile from args.metafile_path, and initialize new metafile
old_metafile_file = open(args.metafile_path, 'r')
old_metafile = json.load(old_metafile_file)
coco_metafile = {"info": {
                    "description": "RoadMotion Coco Dataset","url": "https://clearmotion.com/roadmotion","version": "1.0","year": 2024,"contributor": "Chun Tao","date_created": "2024/03/31"
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {"supercategory": "event","id": 0,"name": "Pothole"},
                    {"supercategory": "event","id": 1,"name": "Manhole Cover"},
                    {"supercategory": "event","id": 2,"name": "Drain Gate"},
                    {"supercategory": "event","id": 3,"name": "Road Crack"},
                    {"supercategory": "event","id": 4,"name": "Speed Bump"}
                ]
                }
frame_count, ann_count = 0, 0

#%% Loop through each event and its frames in the old metafile
for event_id, event_dict in old_metafile.items():
    for frame_count, frame_name in enumerate(event_dict['frames']):
        # Add the frame instance to coco_metafile['images']
        coco_metafile['images'].append({
            "id": frame_count,
            "file_name": frame_name,
            "width": 1920,
            "height": 1080,
            "date_captured": event_id
        })
        # Add the annotation instance to coco_metafile['annotations']
        pdb.set_trace()
        
pdb.set_trace()