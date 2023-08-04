DATASET_JSONFILE_PATH=/home/nano01/a/tao88/cav-mae/process_events/datafiles/frame_csv_data.json
DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26
CAL_DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26
DOWNLOAD_CSV=1
WHEELACCEL_TIME=1.024 # 512 samples 

python3 process_events/train.py \
-j ${DATASET_JSONFILE_PATH} \
-d ${DATA_PATH} \
--cal-data-path ${CAL_DATA_PATH} \
--download-csvs ${DOWNLOAD_CSV} \
--wheelaccel-timespan ${WHEELACCEL_TIME}