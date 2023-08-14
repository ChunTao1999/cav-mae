# DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/frames_rv
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26
PREV_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_8.3.json
JSON_SAVE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_manually_labeled_new.json
EVENT_TYPE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/event_types_manual_label.json

python3 postprocess_data.py \
-d ${DATA_FOLDER_PATH} \
-p ${PREV_JSON_PATH} \
-e ${EVENT_TYPE_PATH} \
-s ${JSON_SAVE_PATH} \
-f 800 450
