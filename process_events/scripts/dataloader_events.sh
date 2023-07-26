DATA_PATH=/home/nano01/a/tao88/shared_folder/CV/events_7.26
CAL_DATA_PATH=/home/nano01/a/tao88/shared_folder/CV/events_7.26
DOWNLOAD_CSV=0

python3 /home/nano01/a/tao88/cav-mae/process_events/dataloader_events.py \
-d ${DATA_PATH} \
--cal-data-path ${CAL_DATA_PATH} \
--download-csvs ${DOWNLOAD_CSV}