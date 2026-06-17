import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pipeline import run_redaction_pipeline, get_whisper_model
from audio_utils import bleep_audio
from config import get_secure_temp_dir, analyzer_engine
from presidio_anonymizer import AnonymizerEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize AnonymizerEngine singleton
anonymizer_engine = AnonymizerEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application.
    Eagerly loads the Whisper model on startup to avoid lazy-loading concurrency issues.
    """
    logger.info("Starting up Redactify API. Eagerly loading Whisper model...")
    try:
        # Load the Whisper model eagerly in a thread pool to avoid blocking the event loop
        await asyncio.to_thread(get_whisper_model)
    except Exception as e:
        logger.error(f"Failed to eagerly load Whisper model on startup: {e}")
    yield
    logger.info("Shutting down Redactify API...")

app = FastAPI(
    title="Redactify API",
    description="FastAPI application to redact/bleep PII from audio files using faster-whisper, microsoft-presidio, and ffmpeg.",
    version="1.0.0",
    lifespan=lifespan
)

def cleanup_files(*filepaths: str):
    """
    Remove temporary files. Used as a FastAPI BackgroundTask or manually in exception handling.
    """
    for path in filepaths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Successfully deleted temporary file: {path}")
        except Exception as e:
            logger.error(f"Failed to delete temporary file {path}: {e}")

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API and dependencies are accessible.
    """
    return {"status": "healthy", "service": "Redactify"}

@app.post("/redact-audio")
async def redact_audio_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    full_redact: bool = False
):
    """
    POST endpoint that accepts an audio file, transcribes it, detects PII,
    bleeps the corresponding segments, and returns the redacted audio file.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
        
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".wav"  # Default fallback if no extension is provided
        
    temp_dir = get_secure_temp_dir()
    input_file_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}{ext}")
    output_file_path = os.path.join(temp_dir, f"redacted_{uuid.uuid4()}{ext}")
    
    logger.info(f"Received file upload (extension: '{ext}'). Saving temporarily to {input_file_path}")
    
    # Save the uploaded file to disk asynchronously
    try:
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        await file.close()
        
    # Schedule file cleanup to run after response is sent
    background_tasks.add_task(cleanup_files, input_file_path, output_file_path)
    
    try:
        # Step 1, 2, 3: Run pipeline to identify PII and get bleep segments
        bleep_segments = await run_redaction_pipeline(input_file_path, full_redact=full_redact)
        
        # Step 4: Run ffmpeg complex filter to bleep out segments
        await bleep_audio(input_file_path, output_file_path, bleep_segments)
        
        if not os.path.exists(output_file_path):
            raise RuntimeError("FFmpeg completed successfully, but the output file was not found on disk.")
            
        # Determine standard media type based on extension
        media_type = "audio/wav"
        lowered_ext = ext.lower()
        if lowered_ext == ".mp3":
            media_type = "audio/mpeg"
        elif lowered_ext in [".m4a", ".mp4"]:
            media_type = "audio/mp4"
        elif lowered_ext == ".ogg":
            media_type = "audio/ogg"
        elif lowered_ext == ".flac":
            media_type = "audio/flac"
            
        logger.info(f"Redaction process completed. Serving file: {output_file_path}")
        
        # Return FileResponse. FastAPI handles the streaming, and the BackgroundTasks
        # trigger cleanups afterward.
        return FileResponse(
            path=output_file_path,
            media_type=media_type,
            filename=f"redacted_{file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Redaction pipeline failed: {e}")
        # Clean up files immediately on failure as response is aborted
        cleanup_files(input_file_path, output_file_path)
        raise HTTPException(status_code=500, detail=f"Redaction process failed: {str(e)}")

async def redact_single_text(text: str, full_redact: bool) -> str:
    """
    Helper to run Presidio Analyzer and Anonymizer on a single block of text.
    """
    if not text.strip():
        return text
    entities = None if full_redact else ["CREDIT_CARD"]
    results = await asyncio.to_thread(
        analyzer_engine.analyze,
        text=text,
        language="en",
        entities=entities
    )
    anonymized_result = await asyncio.to_thread(
        anonymizer_engine.anonymize,
        text=text,
        analyzer_results=results
    )
    return anonymized_result.text

async def redact_json_inplace(data, full_redact: bool) -> None:
    """
    Recursively redact string values within a JSON object in-place.
    Surgically redacts only conversation entry "text" fields if it matches the conversational schema.
    """
    if isinstance(data, dict):
        # Specific check for conversational entries transcript schema
        if "entries" in data and isinstance(data["entries"], list):
            is_conversation = all(
                isinstance(entry, dict) and "text" in entry
                for entry in data["entries"] if isinstance(entry, dict)
            )
            if is_conversation:
                for entry in data["entries"]:
                    if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                        entry["text"] = await redact_single_text(entry["text"], full_redact)
                return

        # Generic recursive dict traversal
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = await redact_single_text(value, full_redact)
            elif isinstance(value, (dict, list)):
                await redact_json_inplace(value, full_redact)
                
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, str):
                data[idx] = await redact_single_text(item, full_redact)
            elif isinstance(item, (dict, list)):
                await redact_json_inplace(item, full_redact)

@app.post("/redact-text")
async def redact_text_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    full_redact: bool = False
):
    """
    POST endpoint that accepts a text or JSON file (e.g. conversational transcript),
    detects PII, anonymizes it (surgically on field-level for conversational JSON),
    and returns the redacted file of the same type.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
        
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".txt"  # Default fallback if no extension is provided
        
    logger.info(f"Received file upload (extension: '{ext}'). Processing in-memory...")
    
    try:
        # Read the uploaded file's content as text
        contents = await file.read()
        text_content = contents.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
    finally:
        await file.close()
        
    temp_dir = get_secure_temp_dir()
    output_file_path = os.path.join(temp_dir, f"redacted_{uuid.uuid4()}{ext}")
    
    # Schedule file cleanup to run after response is sent
    background_tasks.add_task(cleanup_files, output_file_path)
    
    try:
        # Attempt to parse as JSON to support Approach A (Field-Level JSON Redaction)
        is_json = False
        try:
            json_data = json.loads(text_content)
            is_json = True
        except json.JSONDecodeError:
            pass
            
        if is_json:
            logger.info("JSON file structure detected. Performing surgical field-level redaction...")
            await redact_json_inplace(json_data, full_redact)
            redacted_text = json.dumps(json_data, indent=2)
            media_type = "application/json"
        else:
            logger.info("Raw text structure detected. Performing full text redaction...")
            redacted_text = await redact_single_text(text_content, full_redact)
            media_type = "text/plain"
            
        # Write redacted text to the output file
        logger.info(f"Writing redacted content to {output_file_path}...")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(redacted_text)
            
        logger.info(f"Redaction process completed. Serving file: {output_file_path}")
        
        return FileResponse(
            path=output_file_path,
            media_type=media_type,
            filename=f"redacted_{file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Redaction process failed: {e}")
        # Clean up files immediately on failure as response is aborted
        cleanup_files(output_file_path)
        raise HTTPException(status_code=500, detail=f"Redaction process failed: {str(e)}")
