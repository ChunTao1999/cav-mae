for argValue in "$@"; do
    curl -X GET --data '{"unpackOptions": {"excludeUhfd": true}}' --header 'Content-Type: application/json' "https://api.roadmotion.co/api/v0/spiProdSessions/${argValue}" > ~/Downloads/unpackedSession_${argValue}.json
done