SEED=2
EPOCHS=10
DATASET_METAFILE_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final/datafiles/events_cleaned_editted_775.json
DATA_FOLDER_PATH=/home/nano01/a/tao88/RoadEvent-Dataset-Internal-Final

python3 train_11.19.py \
-m ${DATASET_METAFILE_PATH} \
-d ${DATA_FOLDER_PATH} \
--new-splits 0 \
-s ${SEED} \
-e ${EPOCHS}
