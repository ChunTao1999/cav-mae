DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/
CAL_DATA_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/calibrate/
PREV_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/datafiles/events_metafile_with_labels.json
BEV_FRAME_SIZE="600,600"
TRACK_WIDTH=1.664 # in meters

python3 find_min_rect_bbox.py \
-d ${DATA_FOLDER_PATH} \
-c ${CAL_DATA_PATH} \
-p ${PREV_JSON_PATH} \
-w ${TRACK_WIDTH} \
--bev-frame-size ${BEV_FRAME_SIZE}