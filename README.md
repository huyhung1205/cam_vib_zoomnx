# JetsonZoom – Điều khiển camera realtime cho Jetson Orin NX

JetsonZoom là ứng dụng điều khiển camera **realtime** cho **Jetson Orin NX**, thiết kế tách biệt hoàn toàn giữa:

- **Data plane (Streaming)**: nhận/giải mã luồng video qua **RTSP** (OpenCV; trên Jetson ưu tiên pipeline **GStreamer + NVDEC**)
- **Control plane (PTZ/Zoom)**: gửi lệnh điều khiển qua **ONVIF** (SOAP/XML)

Hai kênh kết nối chạy song song để hạn chế tối đa việc lệnh điều khiển ảnh hưởng tới streaming và ngược lại.

## Kiến trúc

### Dual-stack protocol
- **RTSP (cổng 554)**: video stream (tuỳ camera, H.264/H.265)
- **ONVIF (cổng 80/8899 tuỳ thiết bị)**: PTZ/Zoom control qua SOAP/XML

### 3 tầng luồng (thread)
1. **Producer thread (RTSP)**: kéo frame liên tục bằng OpenCV (RTSP URL hoặc pipeline GStreamer/NVDEC)
2. **Main/Display thread**: lấy frame từ queue, hiển thị và nhận input
3. **Worker thread (ONVIF)**: xử lý lệnh zoom theo vận tốc (velocity) + cơ chế STOP bắt buộc

Nguyên tắc đồng bộ: **queue-based** (thread-safe, không chặn nhau).

## Cài đặt nhanh (5 phút)

### Yêu cầu
- **Phần cứng**: Jetson Orin NX (JetPack 5.x; có GStreamer + NVIDIA plugins)
- **Python**: 3.8+
- **Mạng**: camera hỗ trợ **RTSP** và **ONVIF**

### Thoát khỏi venv:
```bash
deactivate
```

### Cài dependencies & cấu hình

**Windows (PowerShell)**
```powershell
cd JetsonZoom
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

copy .env.example .env
notepad .env
```

**Jetson Orin NX (Ubuntu / JetPack 5.x)**
```bash
cd JetsonZoom
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-pyqt5

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

### Chạy lần đầu

```bash
python -m jetson_zoom
```

Mặc định app sẽ mở **giao diện Qt** (thân thiện, có ô nhập RTSP). Nếu muốn dùng giao diện tối giản OpenCV:
`python -m jetson_zoom --ui opencv`

Nếu máy chưa cài được PyQt, app sẽ tự fallback sang UI OpenCV và log cảnh báo.

Hoặc cài editable để có lệnh `jetson-zoom`:

```bash
pip install -e .
jetson-zoom
```

## Điều khiển phím (khi tích hợp input/UI)

Mapping mặc định trong `EventLoop.handle_key_press()`:
- `i`: Zoom in
- `o`: Zoom out
- `s`: Stop
- `q`: Quit

Input/hiển thị mặc định dùng **OpenCV window** (phím `i/o/s/q`). Nếu bạn muốn tích hợp UI khác (PyQt, v.v.), có thể gọi trực tiếp `EventLoop.handle_key_press()`.

## Zoom quang học dễ điều khiển

Trong UI Qt, mặc định bật chế độ **“Giữ nút để zoom”**:
- Nhấn giữ **Zoom In/Zoom Out** để zoom liên tục
- Nhả nút để **STOP** ngay

Nếu bạn chỉ click nhanh (không giữ), app sẽ tự coi là zoom theo “nhịp” (burst): gửi lệnh zoom trong `Thời lượng (ms)` rồi STOP.

Ngoài ra, bạn có thể **lăn con lăn chuột trên khung video** để zoom theo từng “nhịp” (cấu hình bằng `Con lăn: bước (ms)`).

Giải thích nhanh các thông số:
- `Velocity`: vận tốc tương đối của motor zoom (0.10 = chậm, 1.00 = nhanh)
- `Thời lượng (ms)`: thời gian giữ lệnh `ContinuousMove` trước khi gửi `Stop`
- `Zoom.x`: giá trị zoom mà camera báo về qua ONVIF `GetStatus` (thường là giá trị chuẩn hoá, tuỳ camera có thể không có)
- `ZoomCap`: camera có “quảng cáo” hỗ trợ zoom qua ONVIF PTZ hay không (OK/NO/?)

## Tài liệu

- `QUICKSTART.md`: hướng dẫn chi tiết, troubleshooting, ví dụ tích hợp
- `DESIGN.md`: mô tả kiến trúc, thread model, phân tích hiệu năng
- `IMPLEMENTATION.md`: tổ chức code, tiêu chuẩn chất lượng, chiến lược test

## Tính năng chính

- **Streaming tăng tốc phần cứng (Jetson)**: dùng OpenCV + pipeline GStreamer/NVDEC (`STREAM_BACKEND=gst`)
- **Điều khiển PTZ/Zoom (ưu tiên zoom quang học)**: dùng ONVIF PTZ `ContinuousMove/Stop` để điều khiển zoom của camera (không phóng to ảnh bằng phần mềm)
- **Thread-safe**: Producer/Consumer bằng queue, không chặn main thread
- **An toàn**: STOP bắt buộc sau mỗi lệnh di chuyển, kẹp vận tốc (clamp) về dải an toàn
- **Dev-friendly**: type hints, docstrings, logging có màu, cấu hình bằng `.env`

## Cấu hình (`.env`)

Các biến môi trường đang được đọc trong `jetson_zoom/config.py`:

```env
CAMERA_HOST=192.168.1.70
CAMERA_PORT_RTSP=554
CAMERA_PORT_ONVIF=80
CAMERA_USERNAME=admin
CAMERA_PASSWORD=12345

# Tuỳ chọn: override URL (nếu camera dùng path khác)
# CAMERA_RTSP_URL=rtsp://admin:12345@192.168.1.70:554/stream
# CAMERA_ONVIF_URL=http://192.168.1.70:80/onvif/device_service

# Lưu ý: với thư viện `onvif-zeep` hiện tại, ONVIF DeviceMgmt endpoint luôn là `/onvif/device_service`.
# Bạn chỉ cần đúng `CAMERA_HOST` và `CAMERA_PORT_ONVIF`. Biến `CAMERA_ONVIF_URL` chủ yếu để tham khảo/log.
```

## Quản lý nhiều nguồn camera (name → RTSP)

Thay vì nhập RTSP trực tiếp trong `.env`, bạn có thể lưu danh sách camera vào file `JetsonZoom/sources.txt` (có thể commit lên git):

```text
cam1|rtsp://user:pass@192.168.1.70:554/stream
cam2|rtsp://192.168.1.71:554/stream1
```

Lưu ý: nếu RTSP URL có `user:pass@...` thì cân nhắc **không** commit lên git.

Khi chạy `python -m jetson_zoom`, app sẽ hiện menu chọn source từ file này, hoặc cho phép thêm RTSP mới (và tự lưu lại).

Nếu bạn chạy từ IDE và không thấy menu trong terminal, app sẽ tự mở cửa sổ **Source Picker** (OpenCV) để bạn nhập `name` và `rtsp`.

Nếu bạn muốn **luôn luôn** hiện giao diện nhập RTSP khi khởi động:
- Set `SOURCE_PICKER=always` trong `.env`, hoặc
- Chạy `python -m jetson_zoom --picker`

## Lưu cấu hình (tự khôi phục khi mở lại app)

JetsonZoom tự lưu cấu hình gần nhất (host/port/user/pass/rtsp) vào file:
- Windows: `C:\Users\<you>\.jetsonzoom\state.json`
- Linux/Jetson: `/home/<you>/.jetsonzoom/state.json`

Bạn có thể override bằng biến môi trường `STATE_FILE` hoặc tham số `--state-file`.

Tuỳ chọn CLI:
- `--sources-file path/to/sources.txt`
- `--source cam1`
- `--rtsp rtsp://...` (bỏ qua sources file)

Windows (PowerShell): nếu RTSP URL có `&` thì cần bọc trong dấu nháy đơn, ví dụ:
`python -m jetson_zoom --rtsp 'rtsp://user:pass@192.168.1.70:554/stream?channel=1&subtype=0'`

## Các module chính

| Module | Vai trò |
|---|---|
| `jetson_zoom/streams/rtsp_handler.py` | Producer thread: kéo RTSP, decode GPU, đẩy frame vào queue |
| `jetson_zoom/controllers/onvif_client.py` | Worker thread: nhận lệnh zoom, gửi ONVIF/SOAP + STOP an toàn |
| `jetson_zoom/core/event_loop.py` | Main loop: lấy frame, điều phối input, log metrics, shutdown |
| `jetson_zoom/core/continuous_move.py` | API mức cao cho zoom liên tục theo velocity/duration |
| `jetson_zoom/config.py` | Quản lý cấu hình (đọc `.env`, defaults) |

## Hiệu năng (mục tiêu thiết kế)

| Chỉ số | Giá trị kỳ vọng |
|---|---|
| CPU | ~5–10% (GPU xử lý decode) |
| RAM | ~100–150 MB |
| FPS | 30–60 (tuỳ cấu hình) |
| Độ trễ | RTSP < 100ms, ONVIF command < 50ms (phụ thuộc mạng/camera) |

## Cơ chế an toàn

1. **STOP bắt buộc**: mọi lệnh zoom đều gửi STOP sau khi hết thời gian tác động
2. **Kẹp vận tốc**: velocity được giới hạn trong `[0.1, 1.0]`
3. **Drop frame khi đầy queue**: producer không bị block
4. **Shutdown an toàn**: bắt SIGINT/SIGTERM để cleanup
5. **Cô lập lỗi**: lỗi thread được log, hạn chế lan sang thread khác

## `setup.py` dùng để làm gì?

`setup.py` cho phép cài package theo chuẩn Python, và tạo lệnh console:

```bash
pip install -e .
jetson-zoom
```

Entry point trỏ về `jetson_zoom/__main__.py:main()`.

## Ví dụ & “test” nhanh

Xem `examples/demo.py` (5 ví dụ):
1. Tạo config
2. Đồng bộ queue
3. API zoom
4. Error callbacks
5. Đọc metrics trạng thái

Chạy:

```bash
python -m examples.demo
```

## Cấu trúc thư mục

```text
JetsonZoom/
├── jetson_zoom/
│   ├── __main__.py
│   ├── config.py
│   ├── logger.py
│   ├── streams/
│   │   └── rtsp_handler.py
│   ├── controllers/
│   │   └── onvif_client.py
│   └── core/
│       ├── event_loop.py
│       └── continuous_move.py
├── examples/
│   └── demo.py
├── requirements.txt
├── setup.py
├── .env.example
├── QUICKSTART.md
├── DESIGN.md
└── IMPLEMENTATION.md
```

## Troubleshooting

| Vấn đề | Gợi ý |
|---|---|
| Không thấy video | kiểm tra `CAMERA_HOST`, mở cổng RTSP 554, URL stream đúng |
| Zoom không chạy | kiểm tra ONVIF port/credentials, camera có PTZ/Zoom |
| Thiếu `cv2` / OpenCV | Windows: `pip install opencv-python`; Jetson: `sudo apt-get install python3-opencv` |
| `STREAM_BACKEND=gst` không mở được pipeline | thử `STREAM_BACKEND=opencv` hoặc override `GST_PIPELINE_TEMPLATE` đúng codec |
| `ModuleNotFoundError` | chạy `pip install -r requirements.txt` trong venv |

Xem thêm tại `QUICKSTART.md#troubleshooting`.

## License

Dự án phục vụ nghiên cứu điều khiển camera realtime trên Jetson Orin NX.

---

Xem chi tiết:
- Kiến trúc: `DESIGN.md`
- Chuẩn triển khai: `IMPLEMENTATION.md`
- Hướng dẫn nhanh: `QUICKSTART.md`
