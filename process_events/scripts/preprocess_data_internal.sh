CALIBRATE=1
PERSPECTIVE=1
DOWNLOAD_CSV=1
PREPROCESS=0
DATASET_JSONFILE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/datafiles/events_metafile_with_labels.json # change this for a new data folder
EVENTTYPE_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal/datafiles/event_types.json # keep this for preprocessing
DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV
CAL_DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26
DATASET_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal
WHEELACCEL_TIME=1.024 # 256 samples left and right, 512 samples in total (for 500Hz sensor data)

python3 preprocess_data_internal.py \
-c ${CALIBRATE} \
-p ${PERSPECTIVE} \
-j ${DATASET_JSONFILE_PATH} \
-d ${DATA_PATH} \
--preprocess ${PREPROCESS} \
--eventtype-json-path ${EVENTTYPE_JSON_PATH} \
--cal-data-path ${CAL_DATA_PATH} \
--dataset-path ${DATASET_PATH} \
--download-csvs ${DOWNLOAD_CSV} \
--wheelaccel-timespan ${WHEELACCEL_TIME} \
--session-list "75151" "75208" "75211" "75212" "75213" "75214" "75216" "75218" "75219" "75223" "75224" "75225" "75226" "75307" "75308" "75309" "75310" "75316" "75317" "75318" "75319" "75366" "75367" "75413" "75414" "75415" \
--date-list "7.26" "8.3" "8.3" "8.3" "8.4" "8.4" "8.4" "8.6" "8.6" "8.6" "8.6" "8.6" "8.6" "8.12" "8.12" "8.12" "8.12" "8.12" "8.12" "8.12" "8.12" "8.18" "8.18" "8.31" "8.31" "8.31"
