
FILEPATH="$1"
base64 "$FILEPATH" > "${FILEPATH}_base64.txt"
curl -X POST -F "file=@${FILEPATH}_base64.txt" http://localhost:8000/inference
rm "${FILEPATH}_base64.txt"
