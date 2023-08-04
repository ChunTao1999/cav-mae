#!/bin/bash

for argValue in "$@"; do
    wget -O /home/nano01/a/tao88/RoadEvent-shared/CV/session_csvs/session_${argValue}.csv "https://api.roadmotion.co/api/v0/sessionsApi/hfdCsv?id=${argValue}&interpolateGps=0&gpsCorrectionType=0"
done


