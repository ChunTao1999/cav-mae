import json
import pdb

# load the two dicts
dict_list = []
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_7.26.json", 'r') as in_file_1: # auto-labeled or human-labeled
    dict_list.append(json.load(in_file_1))
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.3.json", 'r') as in_file_2:
    dict_list.append(json.load(in_file_2))
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.4.json", 'r') as in_file_3:
    dict_list.append(json.load(in_file_3))
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.6.json", 'r') as in_file_4:
    dict_list.append(json.load(in_file_4))
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.12.json", 'r') as in_file_5:
    dict_list.append(json.load(in_file_5))
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.18.json", 'r') as in_file_6:
    dict_list.append(json.load(in_file_6))
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.31.json", 'r') as in_file_7:
    dict_list.append(json.load(in_file_7))
# pdb.set_trace()


merged_dict = {'data': {}}
for d in dict_list:
    merged_dict['data'].update(d['data'])
with open("/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_9.5.json", 'w') as out_file:
    json.dump(merged_dict, out_file)
# pdb.set_trace()

frame_count = 0
event_count = len(list(merged_dict['data'].keys()))
print(f'Total event count: {event_count}')
for key, value in merged_dict['data'].items():
    frame_count += len(value['frame_paths'])
print(f'Total frame count: {frame_count}')
pdb.set_trace()
