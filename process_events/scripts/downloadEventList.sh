for argValue in "$@"; do
    curl -X GET --data '{"unpackOptions": {"excludeUhfd": true}}' --header 'Content-Type: application/json' "https://api.roadmotion.co/api/v0/sessionsApi/events?id=${argValue}&eventsName=detectedProfileEventList" > /home/nano01/a/tao88/RoadEvent-shared/CV/session_csvs/eventList_${argValue}.json
done