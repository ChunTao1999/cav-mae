# DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/frames_rv
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26/results/frames_rv_annotated
FILENAME=event_93.648_frame_0_at_time_94.663_dist_5.101_rv.png
JSON_SAVE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/labels.json

python3 postprocess_data.py \
-d ${DATA_FOLDER_PATH} \
-n ${FILENAME} \
-s ${JSON_SAVE_PATH}
