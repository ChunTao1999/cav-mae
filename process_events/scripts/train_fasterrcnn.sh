DATASET_METAFILE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/datafiles/events_cleaned_editted_775.json
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final
SEED=2
EPOCHS=30
START_LR=0.001
WEIGHT_DECAY=0.001
GAMMA=0.97

python3 train_fasterrcnn.py \
-m ${DATASET_METAFILE_PATH} \
-d ${DATA_FOLDER_PATH} \
--new-splits 0 \
-s ${SEED} \
-e ${EPOCHS} \
--start-lr ${START_LR} \
--weight-decay ${WEIGHT_DECAY} \
--exp-scheduler-gamma ${GAMMA}

