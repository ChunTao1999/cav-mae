import json
import numpy as np
import pdb
import cv2


# load the two dicts
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile.json", 'r') as in_file_1:
    events_metafile_dict = json.load(in_file_1)

label_count = [0, 0, 0, 0, 0]

frame_error_key_list = []
wheelspec_error_key_list = []
bbox_error_key_list = []
label_error_key_list = []
for key, value in events_metafile_dict['data'].items():
    wheelAccel_spec = np.load(value['wheelAccel_spec_path'])
    if wheelAccel_spec.shape[1] != 31 or np.isnan(wheelAccel_spec).any():
        wheelspec_error_key_list.append(key)
    if len(value['frame_paths']) != len(value['bbox_coords']):
        bbox_error_key_list.append(key)
        print(len(value['frame_paths']), len(value['bbox_coords']))
    for frame_path in value['frame_paths']:
        # frame = cv2.imread(frame_path)
        # if frame is None: frame_error_key_list.append(key)
        frame_idx = int(frame_path.split('/')[-1].split('_')[5])
        try:
            bbox = np.array(value['bbox_coords'][frame_idx])
            if (bbox.shape[0] * bbox.shape[1]) != 8:
                bbox_error_key_list.append(key)
        except:
            bbox_error_key_list.append(key)

    if int(value['event_label']) < 0 or int(value['event_label']) > 4:
        label_error_key_list.append(key)  
    # label_count[int(value['event_label'])] += 1
    

print(frame_error_key_list, wheelspec_error_key_list, bbox_error_key_list, label_error_key_list)
pdb.set_trace()

key_set = set()
for key in frame_error_key_list:
    key_set.add(key)
for key in wheelspec_error_key_list:
    key_set.add(key)
for key in bbox_error_key_list:
    key_set.add(key)
for key in label_error_key_list:
    key_set.add(key)

for key in key_set:
    del events_metafile_dict['data'][key]
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile.json", 'w') as out_file:
    json.dump(events_metafile_dict, out_file)

pdb.set_trace()
