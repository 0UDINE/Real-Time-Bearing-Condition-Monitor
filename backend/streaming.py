"""
Simulated live signal source.

Replays a real CWRU .mat recording as if it were arriving live from a
vibration sensor, a fixed-size chunk at a time, with a real-time delay
between chunks (scaled by `playback_speed`). This lets the dashboard behave
exactly as it would against a real streaming sensor feed, without needing
physical hardware.

To connect a REAL sensor instead of this simulator: replace `SignalReplay`
with a class exposing the same `chunks()` async generator interface, backed
by your DAQ/sensor SDK instead of a .mat file.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

FAULT_SAMPLE_RATE_HZ = 12000

# Standard CWRU shaft speeds by load condition (0-3 hp).
CWRU_LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def load_de_signal(mat_path: str) -> np.ndarray:
    """Load the Drive-End acceleration channel from one CWRU .mat file.
    Mirrors the training-time loader, including the multi-key disambiguation
    for normal-baseline files that carry a leftover extra *_DE_time key."""
    mat_path = Path(mat_path)
    contents = loadmat(mat_path)
    keys = sorted(k for k in contents if k.lower().endswith('_de_time'))
    if not keys:
        raise ValueError(f'No *_DE_time channel found in {mat_path}')
    if len(keys) == 1:
        return np.asarray(contents[keys[0]], dtype=np.float32).reshape(-1)

    file_match = re.search(r'\d+', mat_path.stem)
    if file_match:
        target_id = file_match.group().lstrip('0') or '0'
        for key in keys:
            key_match = re.search(r'\d+', key)
            if key_match and key_match.group().lstrip('0') == target_id:
                return np.asarray(contents[key], dtype=np.float32).reshape(-1)
    return np.asarray(contents[keys[-1]], dtype=np.float32).reshape(-1)


def guess_rpm_from_filename(mat_path: str) -> float | None:
    """Best-effort: CWRU filenames end in _0.._3.mat indicating load condition.
    Returns None if it can't be inferred — caller should fall back to a
    user-supplied RPM (this is what a real tachometer reading replaces)."""
    match = re.search(r'_([0-3])\.mat$', Path(mat_path).name, re.IGNORECASE)
    if match:
        return CWRU_LOAD_RPM[int(match.group(1))]
    return None


class SignalReplay:
    """Streams a .mat recording out in fixed-size chunks, sleeping between
    chunks to simulate real-time sensor arrival."""

    def __init__(self, mat_path: str, rpm: float | None = None, chunk_size: int = 512, playback_speed: float = 1.0):
        self.mat_path = mat_path
        self.signal = load_de_signal(mat_path)
        self.rpm = rpm or guess_rpm_from_filename(mat_path) or CWRU_LOAD_RPM[0]
        self.chunk_size = chunk_size
        self.playback_speed = max(playback_speed, 0.01)
        self._stopped = False

    def stop(self):
        self._stopped = True

    async def chunks(self):
        """Async generator yielding (chunk: np.ndarray, elapsed_seconds: float)."""
        real_seconds_per_chunk = self.chunk_size / FAULT_SAMPLE_RATE_HZ
        sleep_seconds = real_seconds_per_chunk / self.playback_speed
        position = 0
        elapsed = 0.0
        while position + self.chunk_size <= len(self.signal) and not self._stopped:
            chunk = self.signal[position:position + self.chunk_size]
            position += self.chunk_size
            elapsed += real_seconds_per_chunk
            yield chunk, elapsed
            await asyncio.sleep(sleep_seconds)
