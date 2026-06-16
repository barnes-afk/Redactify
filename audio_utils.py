import asyncio
import logging
import os
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

async def get_audio_channels(input_path: str) -> int:
    """
    Checks the number of audio channels in the input file using ffprobe.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            out_str = stdout.decode().strip()
            if out_str.isdigit():
                return int(out_str)
    except Exception as e:
        logger.warning(f"Failed to detect audio channels using ffprobe: {e}")
    return 1

async def extract_mono_channel(
    input_path: str,
    output_path: str,
    channel: str = "left"
) -> None:
    """
    Extracts a single channel from a stereo file into a mono WAV file.
    'left' extracts Front Left (FL), 'right' extracts Front Right (FR).
    """
    pan_arg = "c0=FL" if channel == "left" else "c0=FR"
    args = [
        "-i", input_path,
        "-filter_complex", f"[0:a]pan=mono|{pan_arg}[out]",
        "-map", "[out]",
        "-y", output_path
    ]
    
    logger.info(f"Extracting {channel} channel with args: {args}")
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        error_msg = stderr.decode(errors="replace")
        logger.error(f"FFmpeg channel extraction failed: {error_msg}")
        raise RuntimeError(f"FFmpeg channel extraction failed: {error_msg}")

async def bleep_audio(
    input_path: str,
    output_path: str,
    segments: List[Tuple[float, float]]
) -> None:
    """
    Mutes original audio on segments and overlays a 1000Hz sine wave tone on those segments.
    Supports both mono and stereo (dual-channel) files. For stereo, bleeps only the left
    (customer) channel while keeping the right (agent) channel completely untouched.
    """
    merged_segments = merge_intervals(segments)
    
    if not merged_segments:
        logger.info("No PII segments to bleep. Copying/re-encoding original audio.")
        args = [
            "-i", input_path,
            "-y", output_path
        ]
    else:
        logger.info(f"Bleeping segments: {merged_segments}")
        pii_expr_parts = [
            f"between(t,{start:.3f},{end:.3f})"
            for start, end in merged_segments
        ]
        pii_expr = " + ".join(pii_expr_parts)
        
        channels = await get_audio_channels(input_path)
        if channels == 2:
            logger.info("Stereo file detected. Applying channel splitting, left bleeping, and merging.")
            # Split left and right channels, bleep the left channel, and merge back to stereo
            filter_complex = (
                f"[0:a]channelsplit=channel_layout=stereo[left][right];"
                f"[left]volume=eval=frame:volume='if({pii_expr},0,1)'[muted_left];"
                f"sine=f=1000:r=44100[sine_gen];"
                f"[sine_gen]volume=eval=frame:volume='if({pii_expr},0.15,0)'[gated_sine];"
                f"[muted_left][gated_sine]amix=inputs=2:duration=first[bleeped_left];"
                f"[bleeped_left][right]amerge=inputs=2[out_a]"
            )
        else:
            logger.info("Mono file detected. Applying standard bleep overlay.")
            # Standard mono bleep overlay
            filter_complex = (
                f"[0:a]volume=eval=frame:volume='if({pii_expr},0,1)'[muted_orig];"
                f"sine=f=1000:r=44100[sine_gen];"
                f"[sine_gen]volume=eval=frame:volume='if({pii_expr},0.15,0)'[gated_sine];"
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
