for argValue in "$@"; do
    curl -X GET --data '{"unpackOptions": {"excludeUhfd": true}}' --header 'Content-Type: application/json' "https://api.roadmotion.co/api/v0/spiProdSessions/${argValue}" > /home/nano01/a/tao88/RoadEvent-shared/CV/session_csvs/unpackedSession_${argValue}.json
done