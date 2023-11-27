import cv2
import json
import numpy as np
import os

import pdb


# load the two dicts
data_path = "/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final"
with open(os.path.join(data_path, "datafiles/events_cleaned_editted_775.json"), 'r') as in_file_1:
    events_dict = json.load(in_file_1)

label_count = [0, 0]

wheelAccel_error_set = set()
frame_error_set = set()
bbox_error_set = set()
label_error_set = set()
other_error_set = set(["s_75212_166808", "s_75214_845745", "s_75219_351501", "s_75308_-0623", "s_75318_212869"])

count_frames = 0
for key, value in events_dict.items():
    # if key in other_error_set: # bbox error
    #     del events_dict[key]

    wheelAccel_spec = np.load(os.path.join(data_path, "wheelAccel_seg", key+'.npy'))
    if wheelAccel_spec.shape[1] != 512 or np.isnan(wheelAccel_spec).any():
        wheelAccel_error_set.add(key)
    if len(value['frames']) == 0:
        frame_error_set.add(key)
    if len(value['rv_rot_rect_box']) != len(value['frames']):
        bbox_error_set.add(key)
    # add difficult label for postprocessing
    value['difficult'] = [0] * len(value['frames'])
    count_frames += len(value['frames'])
print(count_frames)
# pdb.set_trace()

key_set = set()
key_set = wheelAccel_error_set.union(frame_error_set, bbox_error_set, label_error_set, other_error_set)
del_agreement = input(f"Found {len(key_set)}/{len(events_dict)} error events, would you like to continue with deleting them and saving a cleaned dict? (Type 'Y' or 'y' to agree): ")
if del_agreement.lower() == 'y':
    print("Continuing with the code...")
    for key in key_set:
        del events_dict[key]
    with open(os.path.join(data_path, "datafiles/events_metafile_with_labels_cleaned.json"), 'w') as out_file:
        json.dump(events_dict, out_file)
    print("Cleaned metafile saved!")
else:
    print("Execution aborted. You did not agree.")

