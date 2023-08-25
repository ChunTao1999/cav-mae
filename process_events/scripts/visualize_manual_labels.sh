META_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile_manually_labeled_8.18.json
CAL_DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26
FRAME_SAVE_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_8.18/results/frames_rv_manually_labeled

python3 visualize_manual_labels.py \
-c ${CAL_DATA_PATH} \
-j ${META_JSON_PATH} \
-s ${FRAME_SAVE_PATH}