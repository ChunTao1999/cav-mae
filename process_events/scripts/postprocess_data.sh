# DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/frames_rv
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final
PREV_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/datafiles/events_cleaned_editted_775.json
JSON_SAVE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/datafiles/events_metafile_with_labels_cleaned_editted.json
EVENT_TYPE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/datafiles/event_types_manual_label.json

python3 postprocess_data.py \
-d ${DATA_FOLDER_PATH} \
-p ${PREV_JSON_PATH} \
-e ${EVENT_TYPE_PATH} \
-s ${JSON_SAVE_PATH} \
-f 960 540
