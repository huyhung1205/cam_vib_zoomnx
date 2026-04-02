# JetsonZoom - Architecture Design Document

## 1. System Architecture Overview

### Dual-Stack Protocol Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                         Camera (192.168.1.70)               │
├─────────────────────────────────────────────────────────────┤
│  RTSP Server (554)      │      ONVIF Server (80/8899)       │
│  - H.264/H.265 stream   │      - SOAP/XML interface         │
└─────────────────────────────────────────────────────────────┘
         ↑                                  ↑
         │ (Video flow)                     │ (Control flow)
         │ (Non-blocking)                   │ (Async commands)
         ↓                                  ↓
┌──────────────────┐            ┌──────────────────┐
│ RTSP Producer    │            │ ONVIF Worker     │
│ Thread           │            │ Thread           │
│ (OpenCV/GStreamer)│            │ (onvif-zeep)     │
└──────────────────┘            └──────────────────┘
         ↓                                  ↑
      Frame Queue                   Command Queue
    (max 30 frames)              (max 10 commands)
         ↓                                  ↑
┌──────────────────────────────────────────────────┐
│          Main Thread (Event Loop)                │
│  - Pull frames & display (30-60 FPS)             │
│  - Handle keyboard/mouse input                   │
│  - Queue zoom commands                           │
│  - Monitor metrics                               │
└──────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Complete Protocol Separation**
   - RTSP for data (streaming)
   - ONVIF for control (commands)
   - No mixing = no interference

2. **Queue-Based Thread Communication**
   - Producer → Frame Queue → Consumer
   - Main → Command Queue → Worker
   - Non-blocking (items dropped if queue full)
   - Thread-safe by design

3. **Hardware Acceleration**
   - NVIDIA NVDEC: H.264 decoding offloaded to GPU
   - CPU focus: Command processing, not video decode
   - GStreamer nvv4l2decoder: Seamless integration

## 2. Thread Model

### Thread 1: RTSP Producer (RTSPStreamHandler)
**Lifecycle**: Long-lived daemon thread

```
Initialize GStreamer Pipeline
        ↓
Create nvv4l2decoder element (GPU acceleration)
        ↓
Connect to camera RTSP stream
        ↓
[Main Loop]
  ├─ _on_new_sample() callback fires
  ├─ Extract frame from GStreamer buffer
  ├─ Create VideoFrame wrapper
  └─ Push to frame_queue (non-blocking, drop if full)
        ↓
[Cleanup] Set pipeline to NULL state
```

**Performance Impact**:
- CPU: ~5-10% (mostly I/O, NVDEC does heavy lifting)
- Memory: ~100-150 MB (GStreamer pipeline + H.264 buffer)
- Latency: ~50-100ms (network + decode)

### Thread 2: Main Thread (EventLoop)
**Lifecycle**: Application lifetime (blocking join)

```
Initialize all systems
        ↓
[Frame Processing Loop (30-60 FPS regulated)]
  ├─ Try: get frame from queue (timeout 10ms)
  │   ├─ If found: render to display
  │   └─ If none: skip (update background)
  │
  ├─ Try: process keyboard input (callback-based)
  │   └─ If 'i' pressed: continuous_mover.zoom_in()
  │
  ├─ Sleep: frame_time_ms - elapsed_ms
  │   (Maintains stable FPS)
  │
  └─ Every 300 frames: log metrics
        ↓
[Cleanup] signal_handler triggers stop()
```

**Design Notes**:
- Must run on main thread for GUI integration (PyQt5, OpenCV)
- Non-blocking frame queue prevents jitter
- Time-budget regulation ensures FPS stability
- Metrics logged every ~300 frames (10 seconds at 30 FPS)

### Thread 3: ONVIF Worker (ONVIFClient)
**Lifecycle**: Long-lived daemon thread

```
Connect to camera ONVIF service
    ├─ zeep WSDL client setup
    ├─ GetProfiles (find PTZ profile)
    └─ Authenticate if username/password set
        ↓
[Main Loop]
  ├─ command_queue.get(timeout=1.0)
  │   
  ├─ If command received:
  │   ├─ Clamp velocity (0.1 - 1.0)
  │   ├─ Send ContinuousMove SOAP call
  │   ├─ time.sleep(duration_ms)
  │   ├─ Send Stop SOAP call [CRITICAL: safety]
  │   └─ Log completion
  │
  └─ If timeout expires: continue loop
        ↓
[Cleanup] Send final STOP command
```

**Design Notes**:
- STOP command is **mandatory** after every move
- Prevents motor runaway if connection lost
- Non-blocking queue prevents main thread slowdown
- One command processed at a time (serialized)

## 3. Queue Design

### Frame Queue (RTSP → Main)
```
┌─────────────────────────────────┐
│     Frame Queue [maxsize=30]    │
├─────────────────────────────────┤
│ Producer: RTSP thread           │
│ Consumer: Main thread           │
│ Item type: VideoFrame dataclass │
├─────────────────────────────────┤
│ Behavior:                       │
│ - Full predicate: drop frame    │
│ - Empty: skip display iteration │
└─────────────────────────────────┘
```

**Why 30 frames?**
- Buffer 1 second of video at 30 FPS
- Network jitter absorbed
- Small enough for low latency
- Not so small that network hiccups cause dropped frames

```python
# Non-blocking push:
try:
    frame_queue.put_nowait(frame)  # Succeed if space
except queue.Full:
    logger.warning("Frame dropped (queue full)")  # Dropped, logged
```

### Command Queue (Main → ONVIF)
```
┌─────────────────────────────────┐
│     Command Queue [maxsize=10]  │
├─────────────────────────────────┤
│ Producer: Main thread (user)    │
│ Consumer: ONVIF worker thread   │
│ Item type: MoveCommand dataclass│
├─────────────────────────────────┤
│ Behavior:                       │
│ - Full: warn & drop command     │
│ - Gets only with timeout        │
└─────────────────────────────────┘
```

**Why 10 commands?**
- Fast enough to respond to user input
- Buffer network delays
- Worker processes 1 per ~500ms (move duration)

## 4. Continuous Move Logic

### Velocity-Based Control Model
```
User presses 'i' (zoom in)
        ↓
continuous_mover.zoom_in(velocity=0.5, duration_ms=500)
        ↓
onvif_client.queue_zoom_command(
    direction=ZoomDirection.IN,
    velocity=0.5,
    duration_ms=500
)
        ↓
[Worker Thread Executes]
├─1. _send_continuous_move(velocity=+0.5)
│   └─ SOAP call: ContinuousMove ProfileToken, Zoom=0.5
│
├─2. time.sleep(0.5)  [500ms]
│   └─ Motor physically moves lens
│
└─3. _send_stop()
    └─ SOAP call: Stop ProfileToken [CRITICAL]
```

### Why This Design?

**Compared to fixed-step zoom:**
```
Fixed-step (BAD):
├─ No smooth control
├─ User presses, gets fixed jump (1x → 2x, fixed step)
└─ Feels robotic

Velocity-based (GOOD):
├─ User controls duration (brief = small zoom, long = big zoom)
├─ Smooth lens movement (physical reality)
└─ Natural feel like analog zoom control
```

### Safety Mechanisms

**1. Mandatory STOP Command**
```python
try:
    self._send_continuous_move(velocity)
    time.sleep(duration_ms)
    self._send_stop()  # ← ALWAYS executed, even on error
except:
    try:
        self._send_stop()  # ← Emergency STOP in except
    except:
        logger.error("Emergency stop FAILED")  # ← Alert!
```

**2. Velocity Clamping**
```python
velocity = max(0.1, min(1.0, velocity))
# Prevents commands like velocity=2.0 (invalid, unsafe)
```

**3. Queue-Based Throttling**
```python
# Can't queue more than 10 commands
# Prevents flooding camera with thousands of zoom commands
```

## 5. Configuration Management

### Layered Configuration

**Layer 1: Defaults** (in code)
```python
@dataclass
class ContinuousMoveConfig:
    pan_velocity: float = 0.5      # Default
    zoom_velocity: float = 0.5     # Default
    move_interval_ms: int = 500    # Default
```

**Layer 2: Environment Variables** (.env file)
```bash
CAMERA_HOST=192.168.1.70
CAMERA_PORT_RTSP=554
```

**Layer 3: JSON File** (optional)
```json
{
  "host": "192.168.1.70",
  "port_rtsp": 554,
  "zoom_velocity": 0.7
}
```

Priority: JSON > Environment > Defaults

## 6. Error Handling Strategy

### Callback-Based Error Reporting
```python
def on_rtsp_error(msg: str):
    logger.error(f"RTSP: {msg}")
    # Could update UI, send alert, etc.

rtsp_handler = RTSPStreamHandler(
    ...,
    error_callback=on_rtsp_error
)
```

Benefits:
- Decoupled: thread doesn't need to know about UI
- Flexible: main app chooses how to handle errors
- Non-blocking: error handling doesn't delay capture

### Per-Thread Try/Except Blocks
```python
def run(self):
    try:
        self._setup()
        while not stop:
            self._process()
    except Exception as e:
        logger.error(f"Thread error: {e}", exc_info=True)
        if self.error_callback:
            self.error_callback(str(e))
    finally:
        self._cleanup()  # ALWAYS runs
```

Key: `finally` block ensures cleanup even on crash

## 7. Performance Budgets

### CPU Time Budget (Jetson Orin NX)
```
Total available: 8 cores @ ~2.4 GHz

Per-thread allocation:
├─ RTSP Producer: ~1 core (mostly GStreamer internal)
├─ ONVIF Worker: ~0.1 core (SOAP is lightweight)
│  (idle most of the time, busy only during move)
├─ Main Thread: ~0.5 core (frame display, input)
└─ Available for user tasks: ~5.4 cores
```

### Memory Budget
```
RTSP Producer:  ~150 MB (GStreamer buffers)
Main Thread:    ~100 MB (frame storage, Qt/CV)
ONVIF Worker:   ~50 MB  (zeep SOAP parser)
Python runtime: ~100 MB
────────────────────────
Total:          ~400 MB (excludes OS, system libraries)
```

### Latency Budget
```
User input (key press)
  ↓ (1 ms)
Main thread event handler
  ↓ (1 ms)
Queue push
  ↓ (10 ms, waits for worker to get command)
Worker thread SOAP call
  ↓ (50-100 ms, network + camera processing)
Camera motor starts moving
────────────────────────
**Total: 60-110 ms** [user notices < 150ms]
```

## 8. Extensibility Points

### 1. Custom Event Handler
```python
class MyGUIHandler(EventHandler):
    def on_key_press(self, key):
        # Custom input handling
        pass
    
    def on_frame_received(self, frame):
        # Custom rendering
        pass

event_loop = EventLoop(..., event_handler=MyGUIHandler())
```

### 2. Custom Configuration
```python
config = ApplicationConfig(
    camera=CameraConfig(
        host="my-camera.local",
        username="user",
        password="pass"
    ),
    streaming=StreamingConfig(
        target_fps=60,
        display_width=2560
    )
)
```

### 3. Error Callbacks
```python
def alert_user(error: str):
    # Send email, SMS, update dashboard, etc.
    pass

rtsp_handler = RTSPStreamHandler(..., error_callback=alert_user)
```

## 9. Testing Strategy

### Unit Tests by Component
```
tests/
├─ test_config.py        # Configuration loading
├─ test_streams.py       # RTSP mocking
├─ test_controllers.py   # ONVIF mocking
├─ test_core.py          # Continuous move logic
└─ test_integration.py   # Full workflow
```

### Mock Objects
- Mock camera RTSP (pre-recorded stream)
- Mock ONVIF server (echo commands)
- Queue inspection (verify commands transmitted)

## 10. Deployment on Jetson Orin NX

### Pre-requisites
```bash
# System libraries
sudo apt-get install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  python3-opencv

# NVIDIA components (JetPack)
# L4T Multimedia API for NVDEC
```

### Virtual Environment
```bash
python3 -m venv .venv --system-site-packages
# ↑ trên Jetson nên dùng system OpenCV/GStreamer packages
source .venv/bin/activate
pip install -r requirements.txt
```

### System Service (optional)
```ini
[Unit]
Description=JetsonZoom Camera Control
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/opt/jetsonzoom
ExecStart=/opt/jetsonzoom/.venv/bin/python -m jetson_zoom
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
