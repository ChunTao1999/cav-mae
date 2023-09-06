CALIBRATE=1
PERSPECTIVE=1
DOWNLOAD_CSV=1
PREPROCESS=1
SEED=4
NUM_EPOCHS=20
RESUME=1
LR=0.001
GAMMA=0.95
DATASET_JSONFILE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/events_metafile.json # change this for a new data folder
EVENTTYPE_JSON_PATH=/home/nano01/a/tao88/RoadEvent-Dataset/datafiles/event_types.json # keep this for preprocessing
DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_8.31
CAL_DATA_PATH=/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26
DATASET_PATH=/home/nano01/a/tao88/RoadEvent-Dataset
MODEL_SAVE_PATH=/home/nano01/a/tao88/cav-mae/process_events/train_results
WHEELACCEL_TIME=1.024 # 512 samples 
MODE=reg_and_cls

python3 train_vggish.py \
-c ${CALIBRATE} \
-p ${PERSPECTIVE} \
-j ${DATASET_JSONFILE_PATH} \
-d ${DATA_PATH} \
-s ${SEED} \
-e ${NUM_EPOCHS} \
--resume-from-checkpoint ${RESUME} \
--preprocess ${PREPROCESS} \
--eventtype-json-path ${EVENTTYPE_JSON_PATH} \
--cal-data-path ${CAL_DATA_PATH} \
--dataset-path ${DATASET_PATH} \
--download-csvs ${DOWNLOAD_CSV} \
--wheelaccel-timespan ${WHEELACCEL_TIME} \
--exp-scheduler-gamma ${GAMMA} \
--model-save-path ${MODEL_SAVE_PATH} \
--train-mode ${MODE}