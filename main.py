import logging
import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pipeline import run_redaction_pipeline
from audio_utils import bleep_audio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Redactify API",
    description="FastAPI application to redact/bleep PII from audio files using faster-whisper, microsoft-presidio, and ffmpeg.",
    version="1.0.0"
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
    file: UploadFile = File(...)
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
        
    temp_dir = tempfile.gettempdir()
    input_file_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}{ext}")
    output_file_path = os.path.join(temp_dir, f"redacted_{uuid.uuid4()}{ext}")
    
    logger.info(f"Received file upload: '{file.filename}'. Saving temporarily to {input_file_path}")
    
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
        bleep_segments = await run_redaction_pipeline(input_file_path)
        
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
