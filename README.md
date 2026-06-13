# Redactify

Redactify is a visually-silent, high-performance FastAPI application built to automatically sanitize sensitive data from audio recordings. It transcribes uploaded audio files, scans the text for PII (Personally Identifiable Information), maps the character indices back to precise word-level audio timestamps, and bleeps out sensitive segments using FFmpeg.

---

## Key Features

* **Precision Word-Level Redaction:** Leverages `faster-whisper`'s word-level timestamps to bleep out only the exact spoken words of sensitive data (instead of muting entire sentences).
* **Robust Spoken Credit Card Detection:** Features a custom, Luhn-validating (`Mod-10`) Microsoft Presidio recognizer that extracts credit card numbers even when spoken digit-by-digit (with arbitrary spaces/hyphens commonly found in spoken transcriptions).
* **Flexible Redaction Modes:**
  * **Default Mode (Credit Cards Only):** Optimized for PCI compliance. Only credit cards are redacted, saving significant CPU cycles by skipping advanced NLP checks for other entity types.
  * **Full Redaction Option:** Optionally redacts all available PII (Names, Phone Numbers, Emails, Social Security Numbers, IP addresses, etc.).
* **Non-Blocking Architecture:** Fully asynchronous operations—FFmpeg processes and CPU-bound ML steps are offloaded to native subprocesses and background threads to keep the event loop highly responsive.
* **Pleasant Acoustic Redaction:** Rather than generating peak digital square waves that can hurt a listener's ears, Redactify mutes the original signal and overlays a soft, comfortable `1000Hz` sine wave tone at `15%` amplitude.
* **Auto-Cleanup Temporary Files:** Integrated FastAPI `BackgroundTasks` ensure that temporary files (uploaded inputs and redacted outputs) are securely deleted from disk immediately after being streamed back to the client.

---

## Architecture Flow

```text
  [ Upload Audio File ]
            │
            ▼
┌──────────────────────────────────────┐
│       faster-whisper (CPU int8)      │ ──► Generates word-level timestamps &
└──────────────────────────────────────┘     constructs a mapped text string
            │
            ▼
┌──────────────────────────────────────┐
│  microsoft-presidio + en_core_web_sm │ ──► Extracts PII character ranges
└──────────────────────────────────────┘     (Default: Credit Card only; Option: Full PII)
            │
            ▼
┌──────────────────────────────────────┐
│          Pipeline Mapping            │ ──► Maps character ranges back to
└──────────────────────────────────────┘     exact audio word start/end timestamps
            │
            ▼
┌──────────────────────────────────────┐
│       ffmpeg complex-filtergraph     │ ──► Mutes audio and overlays a soft,
└──────────────────────────────────────┘     gated 1000Hz sine bleep (15% volume)
            │
            ▼
  [ Stream Redacted File to Client ]
            │
            ▼ (FastAPI Background Task)
  [ Auto-delete Temporary Disk Files ]
```

---

## System Prerequisites

Ensure you have the following installed on your host system:

1. **Python 3.9+** (The project is verified against Python `3.10.12`).
2. **FFmpeg** (The executable binary must be installed and available in your system's `PATH`).
   * **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
   * **macOS (Homebrew):** `brew install ffmpeg`
   * **Windows:** Install FFmpeg via Scoop/Chocolatey or download it from the official website and configure system environment variables.

---

## Installation and Setup

Follow these steps to set up and run Redactify on your machine:

### 1. Install Python Dependencies
Install the required packages in your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Download the NLP Language Model
Download the lightweight English spaCy model (`en_core_web_sm`) used by Presidio Analyzer for basic tokenization and sentence boundaries:
```bash
python3 -m spacy download en_core_web_sm
```

---

## Hosting and Running the API

### Start the FastAPI Server
To host the application locally, run the server using `uvicorn`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Configuration Options (Environment Variables)
You can configure the size of the Whisper model loaded in memory by setting the `WHISPER_MODEL_SIZE` environment variable (defaults to `base`):
* `tiny` (Fastest, lowest memory, ~70MB)
* `base` (Default, balanced, ~140MB)
* `small` (High accuracy, ~460MB)
* `medium` (Very high accuracy, ~1.5GB)

Example of starting with a smaller model:
```bash
export WHISPER_MODEL_SIZE="tiny"
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints and Usage Guide

### 1. Health Check
Verify that the service is online and running.
* **URL:** `/health`
* **Method:** `GET`
* **Response:**
  ```json
  {"status": "healthy", "service": "Redactify"}
  ```

### 2. Redact Audio (POST `/redact-audio`)
Accepts an audio file upload, processes it, and returns the redacted audio file stream.

* **URL:** `/redact-audio`
* **Method:** `POST`
* **Parameters:**
  * `file` (Multipart file upload): The audio file to sanitize.
  * `full_redact` (Boolean Query Parameter, Optional):
    * `false` (default): Only redacts **Credit Card Numbers** (highly optimized).
    * `true`: Redacts **all** PII entities (Names, Phone Numbers, Emails, SSNs, IP addresses, etc.).

#### Important Client/Curl Usage:
When using `curl`, always prefix your file path with an `@` symbol. Without this, `curl` will post the file path as a text string instead of sending the file stream, causing a `422 Unprocessable Entity` validation failure.

#### Example A: Redact Only Credit Cards (Default)
```bash
curl -X POST -F "file=@/home/developer/Downloads/recording.mp3" http://localhost:8000/redact-audio --output redacted_recording.mp3
```

#### Example B: Redact All PII (Full Sanitize)
```bash
curl -X POST -F "file=@/home/developer/Downloads/recording.mp3" "http://localhost:8000/redact-audio?full_redact=true" --output redacted_recording.mp3
```

---

## Running Automated Tests

To run the complete suite of unit tests, integration tests, and actual FFmpeg bleep overlay generation, execute:
```bash
python3 test_app.py
```

The test runner will run **13 comprehensive tests** validating:
* Interval merging mathematical logic.
* Actual FFmpeg filtergraph execution on real temporary audio files.
* Character range mapping back to audio timestamps.
* Custom `LuhnCreditCardRecognizer` regex extraction for spoken credit card digits.
* API endpoints (`/health` and mock file uploads/cleanups).
