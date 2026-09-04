# Bearing Watch --- Real-Time Bearing Condition Monitor

Bearing Watch is a real-time bearing condition-monitoring application
built around a deep-learning model trained on vibration signals from the
**Case Western Reserve University (CWRU) bearing dataset**.

The project combines:

-   a PyTorch bearing-fault classifier,
-   order-tracking preprocessing based on shaft RPM,
-   an ensemble of trained model checkpoints,
-   a FastAPI backend,
-   a WebSocket connection for live updates,
-   and a lightweight browser dashboard.

The current project simulates a live vibration sensor by replaying CWRU
`.mat` recordings chunk by chunk. This makes it possible to test the
complete real-time pipeline without physical sensor hardware.

------------------------------------------------------------------------

## Project structure

``` text
bearing_monitor/
│
├── .venv/                    # Python virtual environment
│
├── backend/
│   ├── main.py              # FastAPI server and real-time session control
│   ├── model.py             # PyTorch model architecture
│   ├── inference.py         # Order tracking and ensemble inference
│   ├── streaming.py         # Simulated live sensor / .mat replay
│   ├── requirements.txt     # Python dependencies
│   └── sample_data/         # CWRU .mat recordings used for replay
│
├── checkpoints/             # Trained .pt model checkpoints
│
├── frontend/
│   └── index.html            # Web dashboard
│
├── notebooks/
│   └── cwru_5channel.ipynb                   # Training / model-development notebook(s)
│
└── README.md
```

------------------------------------------------------------------------

# How the system works

At a high level, the application follows this pipeline:

``` text
CWRU .mat recording
        │
        ▼
  Simulated sensor
  (streaming.py)
        │
        │ chunks of vibration samples
        ▼
   Rolling buffer
      (main.py)
        │
        │ every 2048 new samples
        ▼
  Order tracking
  using actual RPM
        │
        ▼
 Bearing ensemble
    (inference.py)
        │
        ▼
  Deep-learning model
     (model.py)
        │
        ▼
   4-class prediction
        │
        ├── Normal
        ├── Outer race
        ├── Inner race
        └── Ball
        │
        ▼
     WebSocket
        │
        ▼
   Live dashboard
```

## 1. Signal streaming

`backend/streaming.py` loads the **Drive-End (DE) acceleration** signal
from a CWRU `.mat` recording.

Instead of sending the complete recording to the model, it produces
small chunks of samples asynchronously. A delay between chunks makes the
file behave like a vibration sensor producing data in real time.

The playback speed can be increased for demonstrations. For example, a
playback speed of `4` replays the recording four times faster than real
time.

The streaming layer is intentionally separated from the model so that a
real DAQ/sensor source can replace the `.mat` replay later.

------------------------------------------------------------------------

## 2. RPM and order tracking

The vibration signal is recorded at different shaft speeds depending on
the CWRU load condition.

Before inference, the signal is **order-tracked** so that the vibration
representation is normalized toward a reference speed of **1797 RPM**.

``` text
Raw vibration signal
        │
        ▼
Actual shaft RPM
        │
        ▼
Order-track resampling
        │
        ▼
Signal normalized to reference RPM
```

For the current CWRU replay setup, RPM can be inferred from the filename
when it follows the standard CWRU naming convention.

In a real deployment, the RPM should come from a tachometer or motor
controller.

------------------------------------------------------------------------

## 3. Rolling real-time buffer

`backend/main.py` maintains a rolling buffer containing the most recent
vibration samples.

The simulator sends chunks of `512` samples by default, while inference
is triggered after every `2048` newly received samples.

This means the system does **not** run the neural network for every
incoming chunk.

``` text
512 samples  ─┐
512 samples   │
512 samples   ├──► 2048 new samples ──► inference
512 samples  ─┘
```

The backend keeps additional history in the rolling buffer so that each
prediction can use a sufficiently large signal window.

------------------------------------------------------------------------

# The machine-learning model

The model is implemented in `backend/model.py` as:

``` text
EndToEndSTFTClassifier
```

It predicts four classes:

``` text
normal
outer race
inner race
ball
```

The model expects a raw, order-tracked vibration waveform and performs
the signal representation internally.

## Segmenting the signal

Long recordings are divided into overlapping segments.

Current model configuration:

``` text
Segment length : 4096 samples
Segment hop    : 2048 samples
Sample rate    : 12000 Hz
```

Each segment is then transformed into multiple representations.

## Five-channel spectrogram input

For every segment, the model builds five spectrogram channels:

```text
                 Vibration segment
                        │
          ┌─────────────┼───────────────────┐
          │             │                   │
          ▼             ▼                   ▼
      Raw STFT     Full-band        Band-pass envelopes
                   envelope         300–1500 Hz
                      STFT          1500–3000 Hz
                                    3000–5500 Hz
          │             │                   │
          └─────────────┼───────────────────┘
                        ▼
                 5-channel input
                        │
                        ▼
                       CNN
```

The three band-pass channels are produced using 129-tap FIR filters with these frequency ranges:

```text
300–1500 Hz
1500–3000 Hz
3000–5500 Hz
```

These band edges are stored in each notebook-generated checkpoint as `band_edges`. The backend uses them to reconstruct the same multiband filter bank during inference.

The CNN extracts features from all five representations and produces a 128-dimensional feature vector for each segment.

## Attention-based aggregation

A recording can contain many signal segments. Instead of treating every
segment as equally important, the model uses an attention mechanism to
weight the segments before producing the final recording-level
prediction.

Conceptually:

``` text
Segment 1 ──► features ──► attention weight ─┐
Segment 2 ──► features ──► attention weight ─┤
Segment 3 ──► features ──► attention weight ─┼──► final prediction
Segment 4 ──► features ──► attention weight ─┤
    ...                                      ┘
```

This allows the model to give more importance to segments containing
stronger fault signatures.

------------------------------------------------------------------------

# Ensemble inference

`backend/inference.py` loads all `.pt` checkpoints found in the
configured checkpoint directory.

Each checkpoint produces a probability distribution:

``` text
             Normal   Outer   Inner   Ball
Model 1        ...      ...     ...     ...
Model 2        ...      ...     ...     ...
Model 3        ...      ...     ...     ...
```

The probabilities are averaged across the ensemble:

``` text
Individual model probabilities
              │
              ▼
       Probability average
              │
              ▼
       Final probabilities
              │
              ▼
      Highest probability
              │
              ▼
       Predicted class
```

The backend returns both the predicted class and the probabilities for
all four classes.

------------------------------------------------------------------------

# Alert system

The application does not immediately trigger an alert because of a
single non-normal prediction.

By default:

``` text
Required consecutive predictions : 3
Minimum confidence               : 60%
```

An alert therefore requires the same non-normal fault class to be
predicted consecutively with sufficient confidence.

For example:

``` text
Outer race — 72%
Outer race — 81%
Outer race — 75%
        │
        ▼
      ALERT
```

This is intended to reduce false alarms caused by an isolated noisy
prediction.

The thresholds can be changed in `backend/main.py`.

------------------------------------------------------------------------

# Web dashboard

The frontend is a single `index.html` file and does not require a
frontend build system.

The dashboard provides:

-   live Drive-End vibration waveform,
-   current predicted bearing condition,
-   prediction confidence,
-   probabilities for all four classes,
-   confidence history,
-   alert/event log,
-   recording selection,
-   RPM input,
-   playback-speed control,
-   Start/Stop controls.

The browser communicates with the FastAPI backend through a WebSocket
for live prediction updates.

------------------------------------------------------------------------

# Training notebook

The `notebooks/` directory contains the notebook used to train the model
and generate the checkpoints used by the application.

The notebook is kept separate from the deployment code:

``` text
notebooks/
    training notebook
          │
          ▼
    trained checkpoints
          │
          ▼
checkpoints/
          │
          ▼
backend/inference.py
```

The backend does not depend on Jupyter or the notebook at runtime.
`model.py` contains the model architecture needed to reconstruct the
trained network from the saved checkpoints.

> **Important:** if the model architecture or its required configuration
> is changed in the notebook, `backend/model.py` must remain
> synchronized with the architecture used to create the checkpoints.

------------------------------------------------------------------------

# Running the project locally

## Requirements

You need:

-   Python 3.10+ recommended
-   `pip`
-   the project repository
-   trained `.pt` checkpoints
-   CWRU `.mat` recordings for replay

The project can run on CPU. If PyTorch detects a compatible CUDA GPU,
the inference code can use it automatically.

------------------------------------------------------------------------

## 1. Clone the repository

``` bash
git clone https://github.com/0UDINE/Real-Time-Bearing-Condition-Monitor.git
cd bearing_monitor
```

------------------------------------------------------------------------

## 2. Create a virtual environment

### Windows

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should show the virtual environment
name, for example:

``` text
(.venv)
```

------------------------------------------------------------------------

## 3. Install dependencies

From the project root:

``` bash
pip install -r backend/requirements.txt
```

------------------------------------------------------------------------

# 4. Add the model checkpoints

Place the trained `.pt` checkpoint files in:

``` text
bearing_monitor/
└── checkpoints/
    ├── ...
    └── ...
```

The backend searches the configured checkpoint directory for `.pt` files
and loads them as the inference ensemble.

Make sure the checkpoints were generated using an architecture
compatible with `backend/model.py`.

------------------------------------------------------------------------

# 5. Add CWRU recordings

Place the CWRU `.mat` recordings that you want to replay inside:

``` text
backend/sample_data/
```

For example:

``` text
backend/
└── sample_data/
    ├── normal_0.mat
    ├── ...
    └── ...
```

The application automatically searches this directory recursively and
displays the available `.mat` recordings in the dashboard.

------------------------------------------------------------------------

# 6. Start the backend

From the project root:

``` bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see the FastAPI/Uvicorn server start.

Then open:

``` text
http://localhost:8000
```

in your browser.

The FastAPI backend serves the dashboard directly, so you do not need to
start a separate frontend server.

------------------------------------------------------------------------

# 7. Use the dashboard

Once the dashboard is open:

1.  Select a CWRU `.mat` recording.
2.  Optionally enter the RPM manually.
3.  Choose the playback speed.
4.  Click **Start**.
5.  Watch the vibration waveform in real time.
6.  Monitor the predicted condition and confidence.
7.  Check the confidence history.
8.  Check the event log for triggered alerts.

If RPM is left empty, the application attempts to infer it from the
standard CWRU filename convention.

------------------------------------------------------------------------

# Configuration

Several runtime settings are defined in `backend/main.py`:

``` python
PREDICT_EVERY_N_SAMPLES = 2048
BUFFER_MAX_SAMPLES = 4096 * 2

ALERT_STREAK_REQUIRED = 3
ALERT_MIN_CONFIDENCE = 0.60
```

The default directories are:

``` python
CHECKPOINT_DIR = '../checkpoints'
SAMPLE_DATA_DIR = './sample_data'
FRONTEND_DIR = '../frontend'
```

These paths are relative to the `backend` directory when the application
is started using the commands above.

They can also be overridden with environment variables.

### Windows PowerShell example

``` powershell
$env:CHECKPOINT_DIR="../checkpoints"
$env:SAMPLE_DATA_DIR="./sample_data"
$env:FRONTEND_DIR="../frontend"

uvicorn main:app --reload --port 8000
```

------------------------------------------------------------------------

# API and communication

The backend exposes REST endpoints for session management and a
WebSocket for live updates.

Main endpoints include:

``` text
GET  /api/files
GET  /api/session/status
POST /api/session/start
POST /api/session/stop
WS   /ws/live
```

The frontend uses the REST API to control the replay session and uses
`/ws/live` to receive live signal and prediction updates.

------------------------------------------------------------------------

# From simulated sensor to real sensor

The current application uses:

``` text
CWRU .mat file
      ↓
SignalReplay
      ↓
backend
```

A future real deployment can replace the replay source with a real
vibration sensor:

``` text
Vibration sensor
      ↓
DAQ / sensor SDK
      ↓
Signal chunks
      ↓
backend
      ↓
same inference pipeline
      ↓
same dashboard
```

The real sensor implementation should provide the same basic streaming
interface as `SignalReplay`.

A real deployment also needs an actual RPM source, such as a tachometer
or motor controller, because order tracking requires the shaft speed.

------------------------------------------------------------------------

# Known limitations

This project is a prototype for bearing condition monitoring and should
not be treated as a production safety system without additional
validation.

Current limitations include:

-   The model was trained and validated on a specific SKF 6205 bearing
    and test rig.
-   Performance may degrade when used on a different bearing or machine.
-   Ball-fault recall is weaker than the performance for normal,
    inner-race, and outer-race conditions.
-   The simulated streaming source is based on recorded CWRU data rather
    than a physical sensor.
-   The model architecture and configuration must match the checkpoints
    used for inference.

For deployment on different equipment, retraining or fine-tuning with
data from the target machine may be required.

------------------------------------------------------------------------

# Technology stack

``` text
Python
PyTorch
SciPy
NumPy
FastAPI
Uvicorn
WebSockets
HTML / CSS / JavaScript
CWRU Bearing Dataset
```

------------------------------------------------------------------------

# Project workflow

The complete workflow can be summarized as:

``` text
              MODEL DEVELOPMENT
                     │
                     ▼
              Training notebook
                     │
                     ▼
             Trained .pt files
                     │
                     ▼
                checkpoints/
                     │
                     │
                     ▼
              ┌───────────────┐
              │   Backend     │
              │               │
CWRU .mat ───►│ streaming.py  │
              │       │       │
              │       ▼       │
              │ rolling buffer│
              │       │       │
              │       ▼       │
              │ order tracking│
              │       │       │
              │       ▼       │
              │ inference.py  │
              │       │       │
              │       ▼       │
              │   PyTorch     │
              │    model      │
              └───────┬───────┘
                      │
                  WebSocket
                      │
                      ▼
              ┌───────────────┐
              │   Dashboard   │
              │               │
              │ Waveform      │
              │ Prediction    │
              │ Confidence    │
              │ History       │
              │ Alerts        │
              └───────────────┘
```

------------------------------------------------------------------------

## License

Add your project's license information here.
