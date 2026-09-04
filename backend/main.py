"""
FastAPI backend for real-time bearing-fault monitoring.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser — it serves the dashboard
directly and connects to this same backend's WebSocket.
"""
from __future__ import annotations

import asyncio
import glob
import os
import time
from collections import deque
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import BearingEnsemble
from model import ModelConfig
from streaming import SignalReplay

# ---------------------------------------------------------------------------
# Configuration — adjust these paths for your environment.
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '../checkpoints')
SAMPLE_DATA_DIR = os.environ.get('SAMPLE_DATA_DIR', './sample_data')
FRONTEND_DIR = os.environ.get('FRONTEND_DIR', '../frontend')

# How many raw samples must accumulate before we (re-)run inference, and how
# much history to keep in the rolling buffer.
PREDICT_EVERY_N_SAMPLES = 2048
BUFFER_MAX_SAMPLES = 4096 * 2

# Alerting: require this many consecutive non-normal predictions of the same
# class, above this confidence, before raising an alert (avoids one noisy
# segment triggering a false alarm).
ALERT_STREAK_REQUIRED = 3
ALERT_MIN_CONFIDENCE = 0.60

app = FastAPI(title='Bearing Fault Monitor')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

config = ModelConfig()
ensemble: Optional[BearingEnsemble] = None


class SessionState:
    def __init__(self):
        self.task: Optional[asyncio.Task] = None
        self.replay: Optional[SignalReplay] = None
        self.buffer = deque(maxlen=BUFFER_MAX_SAMPLES)
        self.samples_since_predict = 0
        self.recent_labels = deque(maxlen=ALERT_STREAK_REQUIRED)
        self.history = deque(maxlen=500)  # recent predictions, for late-joining clients
        self.events = deque(maxlen=200)   # alert log
        self.running = False
        self.mat_path = None
        self.rpm = None


session = SessionState()
connected_clients: set[WebSocket] = set()


class StartSessionRequest(BaseModel):
    mat_path: str
    rpm: Optional[float] = None
    chunk_size: int = 512
    playback_speed: float = 1.0


@app.on_event('startup')
def load_model():
    global ensemble
    try:
        ensemble = BearingEnsemble(CHECKPOINT_DIR, config)
    except FileNotFoundError as e:
        # Server still starts so you can browse the dashboard / read the
        # error, but predictions will fail until checkpoints are in place.
        print(f'[WARNING] {e}')
        ensemble = None


@app.get('/api/files')
def list_sample_files():
    """Convenience endpoint: lists .mat files available to replay."""
    paths = sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, '**', '*.mat'), recursive=True))
    return {'files': paths}


@app.get('/api/session/status')
def get_status():
    return {
        'running': session.running,
        'mat_path': session.mat_path,
        'rpm': session.rpm,
        'history': list(session.history)[-50:],
        'events': list(session.events)[-50:],
    }


@app.post('/api/session/start')
async def start_session(req: StartSessionRequest):
    if ensemble is None:
        return {'error': f'No model loaded — add checkpoint .pt files to {CHECKPOINT_DIR}'}
    if session.running:
        await stop_session()

    session.replay = SignalReplay(
        req.mat_path, rpm=req.rpm, chunk_size=req.chunk_size, playback_speed=req.playback_speed,
    )
    session.buffer.clear()
    session.samples_since_predict = 0
    session.recent_labels.clear()
    session.running = True
    session.mat_path = req.mat_path
    session.rpm = session.replay.rpm

    session.task = asyncio.create_task(_run_session())
    return {'status': 'started', 'rpm': session.rpm}


@app.post('/api/session/stop')
async def stop_session():
    session.running = False
    if session.replay:
        session.replay.stop()
    if session.task:
        session.task.cancel()
    return {'status': 'stopped'}


async def _run_session():
    start_time = time.time()
    try:
        async for chunk, elapsed in session.replay.chunks():
            session.buffer.extend(chunk.tolist())
            session.samples_since_predict += len(chunk)

            if session.samples_since_predict >= PREDICT_EVERY_N_SAMPLES and len(session.buffer) >= config.segment_length:
                session.samples_since_predict = 0
                await _predict_and_broadcast(chunk, elapsed)
    except asyncio.CancelledError:
        pass
    finally:
        session.running = False
        await _broadcast({'type': 'session_ended'})


async def _predict_and_broadcast(latest_chunk, elapsed_seconds):
    import numpy as np
    buffer_array = np.asarray(session.buffer, dtype=np.float32)

    try:
        prediction = ensemble.predict(buffer_array, rpm=session.rpm)
    except Exception as e:
        await _broadcast({'type': 'error', 'message': str(e)})
        return

    session.recent_labels.append(prediction.label)
    is_streak = (
        len(session.recent_labels) == ALERT_STREAK_REQUIRED
        and all(l == prediction.label for l in session.recent_labels)
        and prediction.label != 'normal'
        and prediction.confidence >= ALERT_MIN_CONFIDENCE
    )

    message = {
        'type': 'update',
        'timestamp': time.time(),
        'elapsed_seconds': round(elapsed_seconds, 2),
        'waveform': latest_chunk[::4].tolist(),  # thin out for lightweight plotting
        'label': prediction.label,
        'confidence': prediction.confidence,
        'probabilities': prediction.probabilities,
        'alert': is_streak,
    }
    session.history.append(message)

    if is_streak:
        event = {
            'timestamp': message['timestamp'],
            'elapsed_seconds': message['elapsed_seconds'],
            'label': prediction.label,
            'confidence': prediction.confidence,
        }
        session.events.append(event)
        await _broadcast({'type': 'alert', **event})

    await _broadcast(message)


async def _broadcast(message: dict):
    stale = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            stale.append(client)
    for client in stale:
        connected_clients.discard(client)


@app.websocket('/ws/live')
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; client doesn't need to send real data
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


# Serve the dashboard at "/" so the whole system runs from one command.
if os.path.isdir(FRONTEND_DIR):
    app.mount('/', StaticFiles(directory=FRONTEND_DIR, html=True), name='frontend')
