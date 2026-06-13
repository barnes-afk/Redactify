import asyncio
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge overlapping and adjacent time intervals.
    """
    if not intervals:
        return []
    
    # Sort intervals by their start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        
        # If current start time is less than or equal to previous end time, merge them
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
            
    return merged

async def bleep_audio(
    input_path: str,
    output_path: str,
    segments: List[Tuple[float, float]]
) -> None:
    """
    Mutes original audio on segments and overlays a 1000Hz sine wave tone on those segments.
    Uses ffmpeg filter_complex with volume evaluation per-frame and amix.
    """
    merged_segments = merge_intervals(segments)
    
    if not merged_segments:
        # If no segments to redact, just copy/re-encode the audio file
        logger.info("No PII segments to bleep. Copying/re-encoding original audio.")
        args = [
            "-i", input_path,
            "-y", output_path
        ]
    else:
        logger.info(f"Bleeping segments: {merged_segments}")
        # Build filter expression for PII segments
        pii_expr_parts = [
            f"between(t,{start:.3f},{end:.3f})"
            for start, end in merged_segments
        ]
        pii_expr = " + ".join(pii_expr_parts)
        
        # Original audio stream: volume is 0 during PII, 1 otherwise
        # Sine wave stream: generated at 1000Hz, volume is 1 during PII, 0 otherwise
        # amix: mixes the original stream and sine wave stream, terminating when original finishes (duration=first)
        filter_complex = (
            f"[0:a]volume=eval=frame:volume='if({pii_expr},0,1)'[muted_orig];"
            f"sine=f=1000:r=44100[sine_gen];"
            f"[sine_gen]volume=eval=frame:volume='if({pii_expr},1,0)'[gated_sine];"
            f"[muted_orig][gated_sine]amix=inputs=2:duration=first[out_a]"
        )
        
        args = [
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[out_a]",
            "-y", output_path
        ]
        
    logger.info(f"Running ffmpeg with args: {args}")
    
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode(errors="replace")
        logger.error(f"FFmpeg failed with return code {process.returncode}. Error: {error_msg}")
        raise RuntimeError(f"FFmpeg execution failed: {error_msg}")
        
    logger.info("FFmpeg bleeping completed successfully.")
