"""RTSP Stream Handler - Producer Thread

Cross-platform RTSP capture thread.

- Windows: typically uses OpenCV + FFmpeg backend automatically.
- Jetson Orin NX: recommended to use OpenCV with CAP_GSTREAMER + an NVDEC pipeline.

This module intentionally avoids importing platform-specific bindings at import time
so the project can be imported on both Windows and Jetson without crashing.
"""

from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any
import sys
import platform
from urllib.parse import urlparse

from jetson_zoom.config import StreamingConfig, CameraConfig
from jetson_zoom.logger import get_logger


@dataclass
class VideoFrame:
    """Represents a single video frame."""

    timestamp: float
    width: int
    height: int
    image: Any  # typically a numpy.ndarray (BGR) from OpenCV

    def __repr__(self) -> str:
        return f"VideoFrame(ts={self.timestamp:.2f}s, {self.width}x{self.height})"


class RTSPStreamHandler(threading.Thread):
    """Producer thread: Acquires RTSP stream and pushes frames to a queue.

    Architecture:
    - Runs in a separate daemon thread
    - Uses OpenCV VideoCapture
    - Supports two input modes:
      - RTSP URL (portable)
      - GStreamer pipeline string (recommended on Jetson for NVDEC)
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        streaming_config: StreamingConfig,
        output_queue: queue.Queue,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize RTSP stream handler.

        Args:
            camera_config: Camera connection settings
            streaming_config: Streaming pipeline settings
            output_queue: Queue to push decoded frames
            error_callback: Optional callback for error messages
        """
        super().__init__(name="RTSPProducer", daemon=True)

        self.logger = get_logger(self.__class__.__name__)
        self.camera_config = camera_config
        self.streaming_config = streaming_config
        self.output_queue = output_queue
        self.error_callback = error_callback

        self._stop_event = threading.Event()
        self._capture: Any = None
        self._opened_event = threading.Event()
        self._open_ok: Optional[bool] = None
        self._last_error: Optional[str] = None

        # Performance tracking
        self._frame_count = 0
        self._dropped_count = 0
        self._last_ok_time = 0.0
        self._start_time = time.time()

    def run(self) -> None:
        """Thread loop: open capture source, read frames, push to queue."""
        try:
            cv2 = self._import_cv2()

            rtsp_url = self.camera_config.build_rtsp_url()
            backend = (self.streaming_config.backend or "auto").strip().lower()
            is_gst = backend in {"gst", "gstreamer", "opencv_gst", "opencv-gst"}

            if backend == "auto":
                # Prefer a GStreamer/NVDEC pipeline on Jetson (Linux + aarch64).
                is_jetson_like = sys.platform.startswith("linux") and platform.machine().lower() in {
                    "aarch64",
                    "arm64",
                }
                is_gst = is_jetson_like

            masked_rtsp = self._mask_url(rtsp_url)
            self.logger.info(f"Opening RTSP source (backend={backend}): {masked_rtsp}")
            self._open_capture(cv2, rtsp_url, prefer_gst=is_gst)
            self._open_ok = True
            self._opened_event.set()

            self._last_ok_time = time.time()

            target_interval_s = 1.0 / max(1, self.streaming_config.target_fps)
            next_frame_t = time.time()

            while not self._stop_event.is_set():
                ok, image = self._capture.read()
                if not ok or image is None:
                    # Brief backoff then retry. If this persists, user likely has a bad URL/network.
                    if time.time() - self._last_ok_time > 5.0:
                        self.logger.warning("No frames received for >5s (check RTSP URL/network).")
                        self._last_ok_time = time.time()
                    time.sleep(0.2)
                    continue

                self._last_ok_time = time.time()

                height, width = image.shape[:2]
                frame = VideoFrame(
                    timestamp=time.time(),
                    width=width,
                    height=height,
                    image=image,
                )

                self._push_frame(frame)
                self._frame_count += 1

                # Soft throttle to requested target FPS (capture may be higher).
                next_frame_t += target_interval_s
                sleep_s = next_frame_t - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_frame_t = time.time()

        except Exception as e:
            error_msg = f"RTSP stream error: {e}"
            self.logger.error(error_msg, exc_info=True)
            self._last_error = error_msg
            self._open_ok = False if self._open_ok is None else self._open_ok
            self._opened_event.set()
            if self.error_callback:
                self.error_callback(error_msg)
        finally:
            self._cleanup()
            self._opened_event.set()

    @staticmethod
    def _mask_url(url: str) -> str:
        try:
            parsed = urlparse(url)
            if not parsed.username and not parsed.password:
                return url
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            user = parsed.username or ""
            auth = f"{user}:***@" if user else "***@"
            path = parsed.path or ""
            query = f"?{parsed.query}" if parsed.query else ""
            return f"{parsed.scheme}://{auth}{host}{port}{path}{query}"
        except Exception:
            return url

    def _open_capture(self, cv2, rtsp_url: str, prefer_gst: bool) -> None:
        candidates: list[tuple[str, str, int]] = []

        if prefer_gst:
            api_preference = getattr(cv2, "CAP_GSTREAMER", 0)
            if not api_preference:
                self.logger.warning(
                    "OpenCV CAP_GSTREAMER is not available; falling back to backend=opencv."
                )
            else:
                codec = (getattr(self.streaming_config, "gst_codec", "auto") or "auto").strip().lower()
                templates: list[str] = []
                if codec == "h264":
                    templates = [self.streaming_config.gst_pipeline_template]
                elif codec == "h265":
                    templates = [self.streaming_config.gst_pipeline_template_h265]
                else:
                    templates = [
                        self.streaming_config.gst_pipeline_template,
                        self.streaming_config.gst_pipeline_template_h265,
                    ]

                seen: set[str] = set()
                for tmpl in templates:
                    if tmpl in seen:
                        continue
                    seen.add(tmpl)
                    candidates.append(("gst", tmpl.format(rtsp_url=rtsp_url), api_preference))

        # Always include a plain RTSP URL fallback.
        candidates.append(("opencv", rtsp_url, 0))

        last_open_error: Optional[str] = None
        for mode, source, api_preference in candidates:
            try:
                cap = (
                    cv2.VideoCapture(source, api_preference)
                    if api_preference
                    else cv2.VideoCapture(source)
                )
                if cap is not None and cap.isOpened():
                    self._capture = cap
                    self.logger.info(f"RTSP capture opened (mode={mode}).")
                    return

                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

                last_open_error = f"open_failed(mode={mode})"
            except Exception as e:
                last_open_error = f"open_error(mode={mode}): {e}"

        self._last_error = (
            "Failed to open video source. "
            "Tried GStreamer (H.264/H.265) and OpenCV fallback. "
            "Common causes: wrong RTSP URL, camera codec mismatch (H.265 vs H.264), "
            "missing GStreamer/NVIDIA plugins on Jetson, or OpenCV without GStreamer."
            + (f" ({last_open_error})" if last_open_error else "")
        )
        raise RuntimeError(self._last_error)

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            detail = str(e).strip()
            tls_hint = ""
            if "static TLS block" in detail and sys.platform.startswith("linux"):
                # Common on Jetson/aarch64 with OpenMP (libgomp) + certain loaders.
                tls_hint = (
                    " Detected 'static TLS block' issue (libgomp/OpenMP). "
                    "Workaround: install libgomp1 and run with LD_PRELOAD, e.g. "
                    "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1"
                )
            raise ImportError(
                "OpenCV (cv2) is required for RTSP capture. "
                "On Windows: pip install opencv-python. "
                "On Jetson: sudo apt-get install python3-opencv. "
                "If you use a virtualenv on Jetson, create it with --system-site-packages "
                "so it can see the apt-installed cv2."
                f"{tls_hint}"
                + (f" (Import error: {detail})" if detail else "")
            ) from e
        return cv2

    def _push_frame(self, frame: VideoFrame) -> None:
        """Push a frame to the output queue without blocking.

        If the queue is full, drop the oldest frame so the display stays 'live'.
        """
        try:
            self.output_queue.put_nowait(frame)
            return
        except queue.Full:
            pass

        try:
            _ = self.output_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.output_queue.put_nowait(frame)
        except queue.Full:
            self._dropped_count += 1

    def stop(self) -> None:
        """Stop the stream handler gracefully."""
        self.logger.info("Stopping RTSP stream...")
        self._stop_event.set()

    def wait_until_opened(self, timeout_s: float) -> bool:
        try:
            self._opened_event.wait(timeout=max(0.0, float(timeout_s)))
        except Exception:
            return False
        return bool(self._open_ok)

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def _cleanup(self) -> None:
        """Release capture resources."""
        try:
            if self._capture is not None:
                self._capture.release()
        except Exception:
            pass

        # Calculate and log performance metrics
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            fps = self._frame_count / elapsed
            self.logger.info(
                f"Stream stopped - Frames: {self._frame_count}, "
                f"Avg FPS: {fps:.1f}, Duration: {elapsed:.1f}s"
            )

    def get_stats(self) -> dict:
        """Get current stream statistics.

        Returns:
            Dictionary with performance metrics
        """
        elapsed = time.time() - self._start_time
        return {
            "frame_count": self._frame_count,
            "dropped_count": self._dropped_count,
            "elapsed_seconds": elapsed,
            "avg_fps": self._frame_count / elapsed if elapsed > 0 else 0,
            "queue_size": self.output_queue.qsize(),
        }
