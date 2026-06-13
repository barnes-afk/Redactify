import asyncio
import logging
from typing import List, Tuple
from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, analyzer_engine

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request
_whisper_model = None

def get_whisper_model() -> WhisperModel:
    """
    Lazy loader for the WhisperModel singleton.
    Loads the model on CPU with int8 quantization for optimal speed and resource usage.
    """
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on CPU (int8)...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully.")
    return _whisper_model

def _transcribe_blocking(audio_path: str) -> Tuple[List, any]:
    """
    Synchronous blocking helper to transcribe and fully consume the segments generator.
    """
    model = get_whisper_model()
    # Request word timestamps for precise bleep segment identification
    segments_gen, info = model.transcribe(audio_path, word_timestamps=True)
    segments_list = list(segments_gen)
    return segments_list, info

async def run_redaction_pipeline(audio_path: str) -> List[Tuple[float, float]]:
    """
    Executes the core redaction pipeline:
    1. Transcribes audio preserving word-level timestamps.
    2. Runs Presidio Analyzer on the transcribed text to find PII character indices.
    3. Maps character indices back to exact audio timestamps.
    Returns a list of (start_time, end_time) tuples representing the bleep intervals.
    """
    logger.info(f"Starting transcription for {audio_path}...")
    try:
        segments, info = await asyncio.to_thread(_transcribe_blocking, audio_path)
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}")
        raise RuntimeError(f"Transcription failed: {e}")

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
    logger.info(f"Transcribed Text: '{cleaned_text}'")
    
    if not cleaned_text:
        logger.info("Transcription is empty. No PII detection needed.")
        return []

    # Run Presidio Analyzer (offloaded to thread pool to keep the event loop responsive)
    logger.info("Running Presidio Analyzer to identify PII...")
    try:
        results = await asyncio.to_thread(
            analyzer_engine.analyze,
            text=full_text,
            language="en"
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
        for w in word_mappings:
            # Overlap exists if word start is before PII end AND word end is after PII start
            if w["start_char"] < result.end and w["end_char"] > result.start:
                bleep_segments.append((w["start_time"], w["end_time"]))

    return bleep_segments
