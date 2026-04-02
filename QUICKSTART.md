# JetsonZoom - Quick Start Guide (Windows + Jetson Orin NX)

## 5-Minute Setup

### 1. Clone & Initialize
```bash
cd JetsonZoom
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd JetsonZoom
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

**Jetson Orin NX (Ubuntu / JetPack 5.x)**
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-pyqt5

cd JetsonZoom
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Camera
```bash
cp .env.example .env
# Edit .env with your camera IP/credentials
nano .env
```

**Example .env:**
```bash
CAMERA_HOST=192.168.1.70
CAMERA_USERNAME=admin
CAMERA_PASSWORD=12345
```

### 2b. Configure RTSP sources (name|rtsp_url)
Edit `sources.txt` (git-friendly):
```text
cam1|rtsp://user:pass@192.168.1.70:554/stream
cam2|rtsp://192.168.1.71:554/stream1
```

Khi chạy app:
- Nếu chạy từ terminal: sẽ có menu chọn source / thêm mới.
- Nếu chạy từ IDE (stdin không interactive): app sẽ mở cửa sổ **Source Picker** (OpenCV).
- Nếu muốn luôn hiện cửa sổ nhập RTSP: set `SOURCE_PICKER=always` hoặc chạy `python -m jetson_zoom --picker`.

### 3. Run Application
```bash
python -m jetson_zoom
```

Mặc định mở UI **Qt** (có ô nhập RTSP). Dùng UI OpenCV tối giản:
```bash
python -m jetson_zoom --ui opencv
```

Nếu PyQt chưa cài/không import được, app sẽ tự fallback sang UI OpenCV và log cảnh báo.

Chạy với RTSP URL trực tiếp (khuyến nghị khi URL dài, có `?` và `&`):
- Windows (PowerShell): nhớ bọc URL trong dấu nháy đơn để tránh PowerShell hiểu `&`
```powershell
python -m jetson_zoom --rtsp 'rtsp://user:pass@192.168.1.70:554/stream?channel=1&subtype=0'
```

You should see:
```
2026-04-01 12:00:00 [INFO] JetsonZoom v1.0.0 - Realtime Camera Control
2026-04-01 12:00:00 [INFO] Step 1/6: Wiring application components...
2026-04-01 12:00:00 [INFO] Step 2/6: Initializing RTSP stream...
2026-04-01 12:00:00 [INFO] Step 3/6: Initializing ONVIF controller...
2026-04-01 12:00:00 [INFO] Step 4/6: Setting up continuous zoom control...
2026-04-01 12:00:00 [INFO] Step 5/6: Creating event loop...
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `i` | Zoom In |
| `o` | Zoom Out |
| `s` | Stop Movement |
| `q` | Quit |

Lưu ý: JetsonZoom gửi lệnh zoom qua **ONVIF PTZ** (thường là **zoom quang học** nếu camera có motor zoom). Ứng dụng không làm “digital zoom” bằng cách crop/scale hình ảnh.

## Mẹo điều khiển zoom dễ hơn (UI Qt)

Trong UI Qt, bật **“Giữ nút để zoom”** để điều khiển giống camera thật:
- Nhấn giữ Zoom In/Out để zoom liên tục
- Nhả nút để dừng ngay

Bạn cũng có thể **lăn con lăn chuột trên khung video** để zoom theo từng bước (mỗi nấc = 1 nhịp, cấu hình bằng `Con lăn: bước (ms)`).

## Understanding the Output

### Log Levels
- **INFO** (🟢): Normal operation steps
- **DEBUG** (🔵): Detailed internals (high frequency)
- **WARNING** (🟡): Frame drops, queue full, etc.
- **ERROR** (🔴): Connection failures, timeouts

### Performance Metrics
Logged every 10 seconds:
```
[INFO] Metrics - FPS: 29.8, Displayed: 300, Dropped: 0 (0.0%), Elapsed: 10.1s
```

**What to watch:**
- `FPS: 29.8` → Should match target (30 FPS in this case)
- `Dropped: 0` → 0 frames lost (good)
- Drop rate `> 5%` → Network congestion or slow camera

## File Structure for Users

```
jetson_zoom/
├── streams/rtsp_handler.py        ← RTSP pull
├── controllers/onvif_client.py    ← Zoom commands
├── core/event_loop.py             ← Main thread
├── config.py                      ← Configuration
└── logger.py                      ← Logging
```

Each module is self-contained:
- Can be tested independently
- Can be extended/customized
- Has full type hints for IDE support

## Common Issues

### Issue 1: OpenCV chưa cài / không import được `cv2`
**Windows:**
```powershell
.\.venv\Scripts\pip install opencv-python
```

**Jetson:**
```bash
sudo apt-get install -y python3-opencv
```

### Issue 2: Dùng `STREAM_BACKEND=gst` nhưng không mở được pipeline
**Nguyên nhân thường gặp:**
- OpenCV build không bật GStreamer
- Pipeline không phù hợp codec (H.265 vs H.264)

**Gợi ý:**
- Thử `STREAM_BACKEND=opencv` (mở trực tiếp RTSP URL)
- Hoặc override `GST_PIPELINE_TEMPLATE` trong `.env` theo đúng codec của camera

### Issue 3: "ONVIF connection failed"
**Check:**
1. Camera IP address is correct
2. Camera is on same network
3. Username/password are correct
4. Camera supports ONVIF (check manual)

**Debug:**
```bash
# Test camera connectivity
ping 192.168.1.70

# Check ONVIF endpoint
curl http://192.168.1.70/onvif/device_service -v
```

### Issue: Bấm zoom nhưng không chạy, log báo queue full / command dropped
**Nguyên nhân thường gặp:** ONVIF chưa kết nối được hoặc camera không hỗ trợ PTZ zoom qua ONVIF.

**Cách kiểm tra nhanh:**
- Nhìn overlay trên cửa sổ: `ONVIF OK/NO`
- Bật `LOG_LEVEL=DEBUG` trong `.env` để xem log kết nối/command chi tiết

### Issue: ONVIF OK nhưng lens không zoom (IMOU/consumer camera)
Một số camera consumer có ống kính 2.8–12mm nhưng **không có motor zoom** (chỉ varifocal chỉnh tay) hoặc chỉ hỗ trợ **digital zoom** trong app của hãng.

Khi đó:
- Lệnh ONVIF `ContinuousMove(Zoom)` có thể bị **ignore** hoặc trả lỗi “Action not supported”.
- JetsonZoom không thể ép zoom quang học nếu camera không expose qua ONVIF.

### Issue 4: "Frame queue full" warnings
**Cause:** Network congestion or slow camera
**Solution:**
- Reduce target FPS in config (30 → 24)
- Check network bandwidth
- Move closer to WiFi router (if wireless)

### Issue 5: High dropped frame rate
**Check in log:**
```
[WARNING] Output queue full, frame dropped
```

**Solutions:**
1. Reduce resolution in config
2. Reduce framerate
3. Check for CPU-heavy background tasks

## Testing Without Camera

### Run Examples
```bash
python -m examples.demo
```

Shows:
- Configuration loading
- Queue synchronization
- Thread creation
- Error handling patterns

### Mock Mode (future)
```python
# In dev branch:
from jetson_zoom.mock import MockCamera

config.camera.mock = True
# Tests all logic without real camera
```

## Code Structure for Developers

### Adding Custom Zoom Velocity
**File:** `.env`
```bash
ZOOM_VELOCITY=0.7  # Default 0.5
```

Or **programmatically:**
```python
config = ApplicationConfig.from_env()
config.continuous_move.zoom_velocity = 0.7
```

### Custom Event Handling
```python
from jetson_zoom.core.event_loop import EventHandler

class MyHandler(EventHandler):
    def on_key_press(self, key):
        if key == "1":
            mover.zoom_in(velocity=0.3)
        elif key == "2":
            mover.zoom_out(velocity=0.3)
    
    def on_frame_received(self, frame):
        # Custom processing
        pass

# Use it:
event_loop = EventLoop(config, mover, rtsp_handler, MyHandler())
```

### Custom Error Handling
```python
def on_error(msg: str):
    print(f"⚠️ {msg}")
    # Send alert, log to file, etc.

rtsp = RTSPStreamHandler(
    camera_config=config.camera,
    streaming_config=config.streaming,
    output_queue=queue.Queue(),
    error_callback=on_error
)
```

## Performance Tips

### 1. Target FPS
```python
config.streaming.target_fps = 24  # Save 20% CPU vs 30
```

### 2. Zoom Speed
```python
config.continuous_move.move_interval_ms = 300  # Faster zooms
config.continuous_move.zoom_velocity = 0.7     # Higher velocity
```

### 3. Queue Sizes
```python
config.streaming.frame_queue_size = 60  # Buffer more = smoother but higher latency
```

## Next Steps

1. **Read DESIGN.md**: Understand architecture (thread model, queues)
2. **Read IMPLEMENTATION.md**: See code organization & patterns
3. **Check examples/demo.py**: Practical usage examples
4. **Customize**: Replace EventHandler with your GUI framework

## Integration with GUI

### PyQt5 Integration Example
```python
from PyQt5.QtCore import QTimer
from jetson_zoom.core.event_loop import EventHandler

class PyQt5Handler(EventHandler, QWidget):
    def on_key_press(self, key):
        # Route to mover
        pass
    
    def on_frame_received(self, frame):
        # Render to QLabel with QPixmap
        pixmap = QPixmap.fromImage(...)
        self.label.setPixmap(pixmap)

# Create timer to call event_loop._process_frame
timer = QTimer()
timer.timeout.connect(lambda: event_loop._process_frame())
```

### OpenCV Integration Example
```python
import cv2

window_name = "JetsonZoom"
cv2.namedWindow(window_name)

while loop_running:
    frame = rtsp_queue.get(timeout=0.01)
    if frame:
        cv2.imshow(window_name, frame.image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('i'):
            mover.zoom_in()
```

## Getting Help

### Check Logs
Most issues logged with full traceback:
```bash
# View real-time logs
python -m jetson_zoom 2>&1 | grep ERROR

# Save to file
python -m jetson_zoom > app.log 2>&1
```

### Debug Mode
Code has logging at DEBUG level (change in logger.py):
```python
logger = get_logger(__name__, level=logging.DEBUG)
```

### Common Debug Checks
```python
# In Python REPL:
from jetson_zoom.config import ApplicationConfig

config = ApplicationConfig.from_env()
print(f"Camera: {config.camera.build_rtsp_url()}")
print(f"ONVIF: {config.camera.build_onvif_url()}")
```

---

**Status**: ✅ Production-ready framework | 🔧 Custom integration needed for GUI
