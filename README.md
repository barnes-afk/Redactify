# Redactify
Utilizing opensource tooling, this project serves to redact sensitive data from audio recordings.


## Running the Application

To start the API in production or development mode:

uvicorn main:app --host 0.0.0.0 --port 8000

You can upload audio files using standard clients or command-line utilities:

curl -X POST -F "file=@sample_recording.mp3" http://localhost:8000/redact-audio --output
redacted_recording.mp3
