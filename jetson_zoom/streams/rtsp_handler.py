"""RTSP Stream Handler - Producer Thread

Cross-platform RTSP capture thread.

- Windows: typically uses OpenCV + FFmpeg backend automatically.
- Jetson Orin NX: recommended to use OpenCV with CAP_GSTREAMER + an NVDEC pipeline.

This module intentionally avoids importing platform-specific bindings at import time
so the project can be imported on both Windows and Jetson without crashing.
"""

import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any
import sys
import platform

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

            source = rtsp_url
            api_preference = 0
            if is_gst:
                source = self.streaming_config.gst_pipeline_template.format(rtsp_url=rtsp_url)
                api_preference = getattr(cv2, "CAP_GSTREAMER", 0)

            self.logger.info(f"Opening RTSP source (backend={backend}): {rtsp_url}")

            self._capture = (
                cv2.VideoCapture(source, api_preference)
                if api_preference
                else cv2.VideoCapture(source)
            )

            if not self._capture.isOpened():
                raise RuntimeError(
                    "Failed to open video source. "
                    "On Jetson, try STREAM_BACKEND=gst and ensure OpenCV has GStreamer enabled. "
                    "Otherwise set CAMERA_RTSP_URL to a valid RTSP URL."
                )

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
            if self.error_callback:
                self.error_callback(error_msg)
        finally:
            self._cleanup()

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OpenCV (cv2) is required for RTSP capture. "
                "On Windows: pip install opencv-python. "
                "On Jetson: sudo apt-get install python3-opencv."
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
