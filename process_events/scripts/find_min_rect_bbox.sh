DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/
PREV_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/datafiles/events_metafile.json

python3 find_min_rect_bbox.py \
-d ${DATA_FOLDER_PATH} \
-p ${PREV_JSON_PATH}