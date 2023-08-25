# DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/frames_rv
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_8.18
PREV_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.18.json
JSON_SAVE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_manually_labeled_8.18_new.json
EVENT_TYPE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/event_types_manual_label.json
FRAMES_PER_EVENT=4

python3 postprocess_data.py \
-d ${DATA_FOLDER_PATH} \
-p ${PREV_JSON_PATH} \
-e ${EVENT_TYPE_PATH} \
-s ${JSON_SAVE_PATH} \
-n ${FRAMES_PER_EVENT} \
-f 800 450
