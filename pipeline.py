import asyncio
import logging
import os
import tempfile
import uuid
from typing import List, Tuple
from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, analyzer_engine, get_secure_temp_dir
from audio_utils import get_audio_channels, extract_mono_channel

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request
_whisper_model = None

def get_whisper_model() -> WhisperModel:
    """
    Lazy loader for the WhisperModel singleton.
    Loads the model on the configured device (e.g. CPU or CUDA/GPU) with the optimal compute type.
    """
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        logger.info("Whisper model loaded successfully.")
    return _whisper_model

def _transcribe_blocking(audio_path: str) -> Tuple[List, any]:
    """
    Synchronous blocking helper to transcribe and fully consume the segments generator.
    Uses highly optimized defaults (beam_size=1, temperature=0.0, language='en', vad_filter=True)
    for maximum processing speed.
    """
    model = get_whisper_model()
    # Request word timestamps for precise bleep segment identification
    segments_gen, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=1,
        temperature=0.0,
        language="en",
        vad_filter=True
    )
    segments_list = list(segments_gen)
    return segments_list, info

async def run_redaction_pipeline(audio_path: str, full_redact: bool = False) -> List[Tuple[float, float]]:
    """
    Executes the core redaction pipeline:
    1. Transcribes audio preserving word-level timestamps.
    2. Runs Presidio Analyzer on the transcribed text to find PII character indices.
       - If full_redact is False (default), only looks for CREDIT_CARD entities.
       - If full_redact is True, looks for all supported PII entities.
    3. Maps character indices back to exact audio timestamps.
    Returns a list of (start_time, end_time) tuples representing the bleep intervals.
    """
    channels = await get_audio_channels(audio_path)
    temp_left_path = None
    transcribe_path = audio_path

    if channels == 2:
        logger.info("Stereo file detected. Extracting Left (Customer) channel for transcription...")
        temp_dir = get_secure_temp_dir()
        temp_left_path = os.path.join(temp_dir, f"temp_left_{uuid.uuid4()}.wav")
        try:
            await extract_mono_channel(audio_path, temp_left_path, "left")
            transcribe_path = temp_left_path
        except Exception as e:
            logger.error(f"Failed to extract Left channel from stereo audio: {e}")
            raise RuntimeError(f"Stereo channel extraction failed: {e}")

    logger.info(f"Starting transcription for {transcribe_path}...")
    try:
        segments, info = await asyncio.to_thread(_transcribe_blocking, transcribe_path)
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        if temp_left_path and os.path.exists(temp_left_path):
            try:
                os.remove(temp_left_path)
                logger.info(f"Successfully deleted temporary Left channel file: {temp_left_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary Left channel file {temp_left_path}: {e}")

    # Build full text and map character indices to word timestamps
    full_text = ""
    word_mappings = []

    for segment in segments:
        if segment.words is not None:
            for word in segment.words:
                word_text = word.word
                start_char_idx = len(full_text)
                full_text += word_text
                end_char_idx = len(full_text)
                
                word_mappings.append({
                    "word": word_text,
                    "start_time": word.start,
                    "end_time": word.end,
                    "start_char": start_char_idx,
                    "end_char": end_char_idx
                })
        else:
            # Fallback to segment-level if words are not available
            segment_text = segment.text
            start_char_idx = len(full_text)
            full_text += segment_text + " "
            end_char_idx = len(full_text)
            
            word_mappings.append({
                "word": segment_text,
                "start_time": segment.start,
                "end_time": segment.end,
                "start_char": start_char_idx,
                "end_char": end_char_idx
            })

    cleaned_text = full_text.strip()
    logger.info(f"Transcription completed successfully. Total characters: {len(cleaned_text)}.")
    
    if not cleaned_text:
        logger.info("Transcription is empty. No PII detection needed.")
        return []

    # Run Presidio Analyzer (offloaded to thread pool to keep the event loop responsive)
    logger.info(f"Running Presidio Analyzer (full_redact={full_redact}) to identify PII...")
    entities = None if full_redact else ["CREDIT_CARD"]
    try:
        results = await asyncio.to_thread(
            analyzer_engine.analyze,
            text=full_text,
            language="en",
            entities=entities
        )
    except Exception as e:
        logger.error(f"Error during Presidio analysis: {e}")
        raise RuntimeError(f"PII analysis failed: {e}")

    # Map PII character ranges back to audio timestamps
    bleep_segments = []
    for result in results:
        logger.info(
            f"Detected PII of type '{result.entity_type}' in character range "
            f"[{result.start}, {result.end}] with score {result.score:.2f}."
        )
        
        # Find all words/segments that overlap with this PII character range
        overlapping_words = [
            w for w in word_mappings
            if w["start_char"] < result.end and w["end_char"] > result.start
        ]
        
        if overlapping_words:
            start_time = min(w["start_time"] for w in overlapping_words)
            end_time = max(w["end_time"] for w in overlapping_words)
            bleep_segments.append((start_time, end_time))

    return bleep_segments
