# JetsonZoom - Implementation Guide

## Project Structure

```
jetson_zoom/
├── __init__.py
├── __main__.py              # Entry point + operational workflow
├── config.py                # Configuration management (5 dataclasses)
├── logger.py                # Colored logging setup
├── streams/
│   ├── __init__.py
│   └── rtsp_handler.py      # Producer thread (OpenCV; tùy chọn GStreamer/NVDEC trên Jetson)
├── controllers/
│   ├── __init__.py
│   └── onvif_client.py      # Worker thread (ONVIF/SOAP)
├── core/
│   ├── __init__.py
│   ├── continuous_move.py   # Zoom velocity-based control
│   └── event_loop.py        # Main thread (display + input)
└── utils/
    └── __init__.py
```

## Code Quality Standards Applied

### 1. Type Hints (PEP 484)
- All functions have explicit parameter and return type hints
- Use Optional, Union, Dict, List from typing module
- Forward references for circular imports

### 2. Docstrings (Google Style)
- Comprehensive docstrings for all modules, classes, methods
- Clear parameter descriptions and return types
- Example usage in docstrings where helpful

### 3. Error Handling
- try/except blocks with specific exception types
- Error callbacks for asynchronous error reporting
- Emergency stop mechanisms (safety critical)
- Proper cleanup in finally blocks

### 4. Thread Safety
- Queue-based inter-thread communication (not shared memory)
- Thread.join() with timeouts for graceful shutdown
- Threading.Event for stop signals
- Non-blocking queue operations (put_nowait, get with timeout)

### 5. Configuration Management
- Dataclass-based configuration (immutable-friendly)
- Environment variable support (.env files)
- JSON file loading capability
- Sensible defaults

### 6. Logging
- Structured logging with context
- Colored output for CLI
- Optional file logging
- Debug/Info/Warning/Error levels

## Core Components

### 1. RTSPStreamHandler (Producer Thread)
**File**: `streams/rtsp_handler.py`

Responsibilities:
- Mở RTSP bằng OpenCV `VideoCapture`
- Trên Jetson: có thể dùng `STREAM_BACKEND=gst` để chạy pipeline GStreamer/NVDEC
- Push frame vào queue (non-blocking) để UI thread hiển thị
- Monitor performance metrics

Key Methods:
- `run()`: Thread main loop
- `stop()`: Graceful shutdown

Safety Features:
- Non-blocking frame push (queue full sẽ drop frame cũ để giữ “live”)
- Reconnect/backoff khi không nhận được frame
- FPS monitoring

### 2. ONVIFClient (Worker Thread)
**File**: `controllers/onvif_client.py`

Responsibilities:
- Establish ONVIF/SOAP connection
- Process zoom commands from queue
- Implement continuous move logic
- Send STOP after every movement

Key Methods:
- `run()`: Thread main loop
- `_connect_onvif()`: Kết nối ONVIF bằng `onvif-zeep` (Media/PTZ services + profile token)
- `_execute_move_command()`: Apply velocity, wait, send STOP
- `queue_zoom_command()`: Queue command (non-blocking)

Safety Features:
- Mandatory STOP command (prevents motor runaway)
- Velocity clamping (0.1-1.0 range)
- Emergency stop on errors
- Command queue prevents overwhelming camera

### 3. ContinuousMover (Zoom Control)
**File**: `core/continuous_move.py`

Responsibilities:
- High-level zoom API
- Manage zoom level state
- Duration-based continuous control
- Non-blocking command submission

Key Methods:
- `zoom_in()`: Queue zoom-in with velocity/duration
- `zoom_out()`: Queue zoom-out
- `stop_movement()`: Emergency STOP
- `get_zoom_level()`: Current position (simulated in demo)

### 4. EventLoop (Main Thread)
**File**: `core/event_loop.py`

Responsibilities:
- Pull frames from RTSP queue
- Handle keyboard/mouse input
- Regulate FPS (frame time limiting)
- Monitor metrics and health

Key Methods:
- `run()`: Main event loop (should run on main thread for GUI)
- `_process_frame()`: Get frame from queue, render
- `_check_metrics()`: Log performance data periodically
- `handle_key_press()`: External input handler

### 5. Configuration
**File**: `config.py`

Five configuration dataclasses:
1. `CameraConfig`: Network, RTSP, ONVIF settings
2. `StreamingConfig`: Video resolution, FPS, GStreamer template
3. `ContinuousMoveConfig`: Velocity ranges, move timing
4. `ApplicationConfig`: Bundled configuration
5. Environment variable support via `from_env()`

## Operational Workflow (6 Steps)

From `__main__.py`:

```
Step 1: Activate venv + Load config
Step 2: Create RTSP Producer thread (GStreamer start)
Step 3: Create ONVIF Worker thread (SOAP connect)
Step 4: Create Continuous Mover wrapper
Step 5: Run Main thread event loop
Step 6: Cleanup on shutdown
```

## Thread Interaction Flow

```
User Input (Keyboard)
        ↓
   Main Thread
        ↓
ContinuousMover.zoom_in()
        ↓
ONVIFClient.queue_zoom_command()
        ↓
Command Queue (max 10 items)
        ↓
Worker Thread: _execute_move_command()
        ├─ Send velocity command
        ├─ Wait (500ms typical)
        └─ Send STOP

Camera
        ↑
GStreamer Pipeline (NVDEC)
        ↑
RTSP Stream
        ↑
Producer Thread: push to Frame Queue
        ↑
Main Thread: display
```

## Testing & Examples

**File**: `examples/demo.py`

Five examples demonstrating:
1. Basic configuration
2. Thread queue synchronization
3. Continuous zoom control
4. Error handling with callbacks
5. Metrics monitoring

Run with:
```bash
python -m examples.demo
```

## Installation & Deployment

### Prerequisites
- Windows 10/11 hoặc Jetson Orin NX (JetPack 5.x)
- Python 3.8+
- (Jetson + `STREAM_BACKEND=gst`) GStreamer 1.0 + NVIDIA plugins (JetPack)
- ONVIF-compatible camera

### Setup
```bash
# Clone/extract project
cd JetsonZoom

# Create virtual environment
# Jetson: nên dùng --system-site-packages để dùng OpenCV từ apt
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure camera
cp .env.example .env
# Edit .env with camera IP/credentials

# Run
python -m jetson_zoom
```

### Key Dependencies
- **zeep**: ONVIF/SOAP client
- **lxml**: XML processing for ONVIF
- **python-dotenv**: Environment configuration
- **OpenCV (cv2)**: RTSP capture + display (Windows: pip, Jetson: apt `python3-opencv`; tuỳ build có thể dùng CAP_GSTREAMER)

## Performance Characteristics

### Target Metrics
- Video display: 30-60 FPS (configurable)
- Zoom command latency: < 100ms (queue to camera)
- CPU usage: ~10-20% (NVDEC offloads H.264)
- Memory: ~200-300 MB (GStreamer + Python runtime)

### Optimization Points
- NVIDIA nvv4l2decoder offloads H.264→raw conversion to GPU
- Non-blocking queues prevent main thread blocking
- Frame dropping on queue overflow (logged as warning)
- Efficient SOAP calls via zeep

## Security Notes

1. **Authentication**: Optional basic auth (ONVIF WSSE)
2. **Network**: Ensure camera on trusted network
3. **.env Protection**: Add to .gitignore (contains passwords)
4. **RTSP**: Plain auth - consider SSH tunneling for production

## Known Limitations

1. **GStreamer Setup**: Requires system GStreamer libraries
2. **ONVIF Compatibility**: Different cameras may need WSDL adjustments
3. **Zoom Range**: Hardcoded 1.0-30.0 (adjust in config for your camera)
4. **GUI Framework Integration**: EventLoop designed for PyQt5/OpenCV integration

## Future Enhancements

- [ ] Multi-camera support
- [ ] Recording capability (via GStreamer filesink)
- [ ] Web API for remote control
- [ ] Focus control (auto-focus after zoom)
- [ ] Camera preset management
- [ ] Telemetry logging (zoom, pan, tilt history)
