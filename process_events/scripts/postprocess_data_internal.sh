# DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/frames_rv
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/
PREV_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/datafiles/events_metafile.json
JSON_SAVE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/datafiles/events_metafile_manually_editted.json
EVENT_TYPE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Source/datafiles/event_types_manual_label.json
FRAMES_PER_EVENT=4

python3 postprocess_data_internal.py \
-d ${DATA_FOLDER_PATH} \
-p ${PREV_JSON_PATH} \
-e ${EVENT_TYPE_PATH} \
-s ${JSON_SAVE_PATH} \
-n ${FRAMES_PER_EVENT} \
-f 800 450
