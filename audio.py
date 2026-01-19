"""Audio decoding utilities using FFmpeg."""

import subprocess
from pathlib import Path

import numpy as np

from config import FFMPEG_PATH, SAMPLE_RATE


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        proc = subprocess.run(
            [FFMPEG_PATH, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def decode_webm_to_pcm_f32(input_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a single .webm file to float32 mono PCM."""
    if not check_ffmpeg_available():
        raise RuntimeError(
            f"ffmpeg not found at '{FFMPEG_PATH}'. "
            f"Please install ffmpeg or set FFMPEG_PATH environment variable. "
            f"Install with: sudo apt install ffmpeg (Ubuntu/Debian) or brew install ffmpeg (macOS)"
        )

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        print("[ffmpeg stderr]\n", stderr)
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr[:200]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio


def decode_webm_bytes_to_pcm_f32(data: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode WebM bytes to float32 mono PCM using ffmpeg (in-memory, no temp files)."""
    if not data:
        return np.zeros(0, dtype=np.float32)

    if len(data) < 4:
        print(f"[decode] Warning: WebM data too short: {len(data)} bytes")
        return np.zeros(0, dtype=np.float32)

    # Check if it looks like WebM (starts with EBML header)
    webm_markers = [b'\x1a\x45\xdf\xa3', b'webm', b'WEBM']
    has_webm_header = any(data.startswith(marker) for marker in webm_markers)
    if not has_webm_header:
        print(f"[decode] Warning: Data doesn't appear to be WebM (first 20 bytes: {data[:20].hex()})")

    if not check_ffmpeg_available():
        raise RuntimeError(
            f"ffmpeg not found at '{FFMPEG_PATH}'. "
            f"Please install ffmpeg or set FFMPEG_PATH environment variable. "
            f"Install with: sudo apt install ffmpeg (Ubuntu/Debian) or brew install ffmpeg (macOS)"
        )

    # Use pipe for input - no temp file needed
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-f", "webm",        # Specify input format explicitly
        "-i", "pipe:0",      # Read from stdin
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",            # Write to stdout
    ]
    proc = subprocess.run(
        cmd,
        input=data,          # Pass data via stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        print(f"[ffmpeg stderr]\n{stderr}")
        if "Invalid data found" in stderr or "moov atom not found" in stderr:
            print("[decode] WebM file appears incomplete or corrupted")
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr[:200]}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio
