for argValue in "$@"; do
    curl -X GET --data '{"unpackOptions": {"excludeUhfd": true}}' --header 'Content-Type: application/json' "https://api.roadmotion.co/api/v0/sessionsApi/events?id=${argValue}&eventsName=detectedProfileEventList" > ~/Downloads/eventList_${argValue}.json
done