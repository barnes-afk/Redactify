# Redactify

Redactify is a visually-silent, high-performance FastAPI application built to automatically sanitize sensitive data from audio recordings. It transcribes uploaded audio files, scans the text for PII (Personally Identifiable Information), maps the character indices back to precise word-level audio timestamps, and bleeps out sensitive segments using FFmpeg.

---

## Key Features

* **Precision Word-Level Redaction:** Leverages `faster-whisper`'s word-level timestamps to bleep out only the exact spoken words of sensitive data (instead of muting entire sentences).
* **Robust Spoken Credit Card Detection:** Features an enterprise-grade custom Presidio recognizer designed for live-agent call transcripts:
    * **Asynchronous Backtracking DFS Scan:** Recursively scans extracted digit sequences allowing conversational pauses of up to `15` tokens (e.g. *"Hold on, let me look at it"*), seamlessly skipping filler words or count adjectives (such as *"four"* in *"last four digits"*).
    * **Density-Quality Greedy Selection:** Greedily evaluates overlapping candidate subsequences, prioritizing candidates with the smallest maximum gap and highest digit density. This prevents false-positive combinations.
    * **Prefix Verification Filter:** Cross-references Luhn-valid digit sequences against known credit card brand standards (Visa, Mastercard, Amex, Discover, Diners Club, JCB) and common test patterns (prefixes `1`–`6`). This drops false positives from spoken prices (e.g. lists of product costs) to virtually zero.
* **Stereo Dual-Channel Audio Support:** Designed for call center stereo recordings where the **Customer** is on the Left channel and the **Agent** is on the Right channel:
    * **Targeted Transcription:** Automatically splits channels and transcribes **only the customer channel (Left)**, saving 50% ML compute overhead (since agents rarely read out their own PII).
    * **Stereo-Preserving Bleeping:** Processes the channels independently and merges them back with `amerge`—meaning customer PII is bleeped on the Left while the Agent is left completely untouched, preserving pristine stereo channel separation for quality assurance (QA) audits.
* **PCI-DSS Compliant RAM Storage (Zero-Disk Footprint):** To comply with strict PCI-DSS regulations, unredacted raw audio containing sensitive PII never touches non-volatile physical storage:
    * **In-Memory RAM Disk:** Automatically writes all uploaded inputs, temporary split channels, and redacted outputs strictly to an in-memory virtual memory RAM disk (`/dev/shm` on Linux).
    * **Secure Fallback:** Falls back to standard temporary storage on unsupported OSes (or local development) while keeping clean asynchronous file garbage collection active.
* **Flexible Redaction Modes:**
    * **Default Mode (Credit Cards Only):** Optimized for PCI compliance. Only credit cards are redacted, saving significant CPU cycles by skipping advanced NLP checks for other entity types.
    * **Full Redaction Option:** Optionally redacts all available PII (Names, Phone Numbers, Emails, Social Security Numbers, IP addresses, etc.).
* **Non-Blocking Architecture:** Fully asynchronous operations—FFmpeg processes and CPU-bound ML steps are offloaded to native subprocesses and background threads to keep the event loop highly responsive.
* **Pleasant Acoustic Redaction:** Rather than generating peak digital square waves that can hurt a listener's ears, Redactify mutes the original signal and overlays a soft, comfortable `1000Hz` sine wave tone at `15%` amplitude.
* **Auto-Cleanup Temporary Files:** Integrated FastAPI `BackgroundTasks` ensure that temporary files (uploaded inputs and redacted outputs) are securely deleted from disk immediately after being streamed back to the client.
* **Structured Chat Transcript & JSON Redaction:** Supports both plain text (`.txt`) and structured chat transcripts (`.json`) with intelligent formatting:
    * **Surgical Field-Level JSON Redaction**: If JSON structure is detected, it automatically parses it. For standard conversational JSON files (containing `"entries"` arrays), it selectively redacts *only* the dialogue `"text"` fields. All structural metadata, including timestamps, IDs, and roles, are preserved perfectly.
    * **Generic JSON Traversal**: For any other arbitrary JSON formats, it recursively parses and redacts *all* string values in-place while keeping keys and document hierarchies intact.
    * **Raw Text Support**: For standard `.txt` files, it sanitizes full-text documents and returns the cleaned file.

---

## Architecture Flow

```text
                  [ Upload Audio File ]
                            │
                            ▼ (Saves strictly in RAM /dev/shm)
             [ Detect Audio Channel Count ]
              /                          \
       (Stereo: 2 Channels)         (Mono: 1 Channel)
             /                            \
┌───────────────────────────┐      ┌───────────────────────────┐
│ Extract Left (Customer)   │      │   Transcribe Full Track   │
│ Channel to Mono Temp WAV  │      │   (audio_path directly)   │
└───────────────────────────┘      └───────────────────────────┘
             │                            │
             ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  faster-whisper (CPU int8)                   │
│   ──► Generates word-level timestamps & constructions        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│             microsoft-presidio + en_core_web_sm              │
│   ──► Extracts PII character ranges from customer channel    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                       Pipeline Mapping                       │
│   ──► Maps character ranges back to word start/end timestamps│
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  ffmpeg complex-filtergraph                  │
│   ──► Stereo: splits FL/FR, bleeps FL, merges back via amerge│
│   ──► Mono: standard soft gated 1000Hz sine bleep overlay     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
            [ Stream Redacted File to Client ]
                            │
                            ▼ (FastAPI Background Task)
         [ Auto-purge Virtual Files from /dev/shm ]
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
You can customize Redactify's performance, resource usage, and hardware acceleration using the following environment variables:

1. **`WHISPER_MODEL_SIZE`**: Specifies the size of the OpenAI Whisper model loaded into memory (defaults to `base`):
    * `tiny` (Fastest, lowest memory usage, ~70MB)
    * `base` (Default, balanced speed/accuracy, ~140MB)
    * `small` (High accuracy, ~460MB)
    * `medium` (Very high accuracy, ~1.5GB)

2. **`WHISPER_DEVICE`**: Specifies the hardware device on which to execute the model (defaults to `cpu`):
    * `cpu` (Default, runs on system CPU)
    * `cuda` (Runs on Nvidia GPU using CUDA for maximum throughput)

3. **`WHISPER_COMPUTE_TYPE`**: Specifies the mathematical quantization/precision type. By default, it automatically selects the most optimal option:
    * **On CPU:** Defaults to `int8` (quantized 8-bit integers, reducing CPU load and RAM usage).
    * **On GPU (`cuda`):** Defaults to `float16` (native half-precision, taking advantage of GPU tensor cores). Can also be configured to `int8_float16` to save VRAM.

#### Execution Examples:

* **Example A: High-Efficiency CPU execution with the small model (default quant `int8`)**
  ```bash
  export WHISPER_MODEL_SIZE="small"
  export WHISPER_DEVICE="cpu"
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

* **Example B: Ultra-Fast GPU execution with the base model**
  ```bash
  export WHISPER_DEVICE="cuda"
  export WHISPER_COMPUTE_TYPE="float16"
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

---

## Running with Docker

Redactify can be fully containerized using Docker, removing the need to manually install system-level packages (like FFmpeg) and Python dependencies on your host machine.

### Docker Compose (Recommended)

To build and start the application instantly with Docker Compose, run:

```bash
docker compose up --build
```

This will automatically:
1. Build the Docker image.
2. Download system dependencies (`ffmpeg`, `libgomp1`).
3. Set up Python package requirements.
4. Pre-download/cache the English spaCy model (`en_core_web_sm`).
5. Pre-warm and cache the default `faster-whisper` model (`base`) within the image for instant startup.
6. Allocate a proper size of `/dev/shm` (in-memory RAM disk) for secure PCI-DSS compliant temporary file processing.
7. Expose the API at `http://localhost:8000`.

### Manual Docker Build & Run

If you prefer to work with raw Docker commands:

#### 1. Build the image
```bash
docker build -t redactify:latest .
```

#### 2. Run the container
```bash
docker run -p 8000:8000 --shm-size=256mb --name redactify-api redactify:latest
```

*Note: The `--shm-size=256mb` flag allocates an in-memory RAM disk inside the container namespace, allowing Redactify to perform high-speed, PCI-DSS compliant audio processing without writing unredacted customer PII to physical storage.*

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

### 3. Redact Text/JSON (POST `/redact-text`)
Accepts a plain text file (`.txt`) or structured JSON file (`.json`) containing call/chat transcripts, sanitizes it, and returns the redacted file of the identical type.

* **URL:** `/redact-text`
* **Method:** `POST`
* **Parameters:**
    * `file` (Multipart file upload): The text or JSON file to sanitize.
    * `full_redact` (Boolean Query Parameter, Optional):
        * `false` (default): Only redacts **Credit Card Numbers** (highly optimized).
        * `true`: Redacts **all** PII entities (Names, Phone Numbers, Emails, SSNs, etc.).

#### Features:
* **Surgical JSON Handling**: If a JSON file containing a conversational `"entries"` structure is uploaded, Redactify surgically redacts *only* the `"text"` fields inside each entry. Timestamps, user IDs, and role labels remain fully untouched to prevent breaking downstream integrations.
* **Generic JSON Fallback**: For other JSON schemas, it recursively redacts *all* string values in-place.
* **Plain Text Support**: For plain `.txt` files, the entire text document is redacted.

#### Example A: Redact Only Credit Cards in Conversational JSON
```bash
curl -X POST -F "file=@chat_transcript.json" http://localhost:8000/redact-text --output redacted_transcript.json
```

#### Example B: Redact All PII in Raw Text File
```bash
curl -X POST -F "file=@transcript.txt" "http://localhost:8000/redact-text?full_redact=true" --output redacted_transcript.txt
```

---

## Running Automated Tests

To run the complete suite of unit tests, integration tests, and actual FFmpeg bleep overlay generation, execute:
```bash
python3 test_app.py
```

The test runner will run **29 comprehensive tests** validating:
* Interval merging mathematical logic.
* Actual FFmpeg filtergraph execution on real temporary audio files.
* Character range mapping back to audio timestamps.
* Custom `LuhnCreditCardRecognizer` regex extraction for spoken credit card digits.
* API endpoints (`/health`, `/redact-audio`, `/redact-text` text/JSON uploads, and mock file cleanups).