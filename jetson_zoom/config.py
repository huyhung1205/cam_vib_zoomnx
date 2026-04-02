"""Configuration management for JetsonZoom."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import os
from dotenv import load_dotenv


def _getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class CameraConfig:
    """Camera connection configuration."""

    # Network
    host: str
    port_rtsp: int = 554
    port_onvif: int = 80
    username: str = ""
    password: str = ""

    # RTSP
    rtsp_url: Optional[str] = None
    rtsp_timeout: int = 10

    # ONVIF
    onvif_url: Optional[str] = None
    onvif_timeout: int = 5

    def build_rtsp_url(self) -> str:
        """Build RTSP URL from components.

        Returns:
            Complete RTSP URL
        """
        if self.rtsp_url:
            return self.rtsp_url

        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{auth}{self.host}:{self.port_rtsp}/stream"

    def build_onvif_url(self) -> str:
        """Build ONVIF URL from components.

        Returns:
            Complete ONVIF URL
        """
        if self.onvif_url:
            return self.onvif_url

        return f"http://{self.host}:{self.port_onvif}/onvif/device_service"

    @classmethod
    def from_env(cls) -> "CameraConfig":
        """Load configuration from environment variables.

        Returns:
            CameraConfig instance
        """
        load_dotenv()
        return cls(
            host=os.getenv("CAMERA_HOST", "192.168.1.100"),
            port_rtsp=int(os.getenv("CAMERA_PORT_RTSP", "554")),
            port_onvif=int(os.getenv("CAMERA_PORT_ONVIF", "80")),
            username=os.getenv("CAMERA_USERNAME", ""),
            password=os.getenv("CAMERA_PASSWORD", ""),
            rtsp_url=os.getenv("CAMERA_RTSP_URL"),
            onvif_url=os.getenv("CAMERA_ONVIF_URL"),
        )

    @classmethod
    def from_file(cls, filepath: Path) -> "CameraConfig":
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON config file

        Returns:
            CameraConfig instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class StreamingConfig:
    """Streaming and display settings."""

    # Display
    target_fps: int = 30
    display_width: int = 1920
    display_height: int = 1080
    frame_queue_size: int = 30

    # Backends
    # - auto: prefer GStreamer pipeline when available, otherwise fallback
    # - opencv: cv2.VideoCapture(rtsp_url) (FFmpeg on Windows, depends on build on Linux)
    # - gst: cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER) (recommended on Jetson)
    backend: str = "auto"
    display_backend: str = "opencv"  # opencv|none
    window_name: str = "JetsonZoom"

    # GStreamer pipeline
    gst_pipeline_template: str = (
        "rtspsrc location={rtsp_url} latency=0 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false"
    )

    @classmethod
    def from_env(cls) -> "StreamingConfig":
        """Load streaming/display configuration from environment variables."""
        backend = os.getenv("STREAM_BACKEND", "auto").strip().lower()
        display_backend = os.getenv("DISPLAY_BACKEND", "opencv").strip().lower()
        window_name = os.getenv("WINDOW_NAME", "JetsonZoom")
        gst_pipeline_template = os.getenv("GST_PIPELINE_TEMPLATE")

        config = cls(
            target_fps=_getenv_int("TARGET_FPS", 30),
            display_width=_getenv_int("DISPLAY_WIDTH", 1920),
            display_height=_getenv_int("DISPLAY_HEIGHT", 1080),
            frame_queue_size=_getenv_int("FRAME_QUEUE_SIZE", 30),
            backend=backend if backend else "auto",
            display_backend=display_backend if display_backend else "opencv",
            window_name=window_name,
        )

        if gst_pipeline_template:
            config.gst_pipeline_template = gst_pipeline_template

        return config


@dataclass
class ContinuousMoveConfig:
    """Continuous PTZ move configuration."""

    # Velocity settings (0.1 - 1.0)
    pan_velocity: float = 0.5
    tilt_velocity: float = 0.5
    zoom_velocity: float = 0.5

    # Interval for continuous move (milliseconds)
    move_interval_ms: int = 500
    move_timeout_s: int = 10

    # Zoom specific
    zoom_min: float = 1.0
    zoom_max: float = 30.0

    @classmethod
    def from_env(cls) -> "ContinuousMoveConfig":
        """Load continuous move configuration from environment variables."""
        return cls(
            pan_velocity=_getenv_float("PAN_VELOCITY", 0.5),
            tilt_velocity=_getenv_float("TILT_VELOCITY", 0.5),
            zoom_velocity=_getenv_float("ZOOM_VELOCITY", 0.5),
            move_interval_ms=_getenv_int("MOVE_INTERVAL_MS", 500),
            move_timeout_s=_getenv_int("MOVE_TIMEOUT_S", 10),
            zoom_min=_getenv_float("ZOOM_MIN", 1.0),
            zoom_max=_getenv_float("ZOOM_MAX", 30.0),
        )


@dataclass
class ApplicationConfig:
    """Main application configuration."""

    camera: CameraConfig
    streaming: StreamingConfig
    continuous_move: ContinuousMoveConfig

    # Threading
    queue_size: int = 100
    producer_thread_name: str = "RTSPProducer"
    worker_thread_pool_size: int = 4

    # Error handling
    max_reconnect_attempts: int = 5
    reconnect_delay_s: int = 2

    @classmethod
    def from_env(cls) -> "ApplicationConfig":
        """Load full application config from environment.

        Returns:
            ApplicationConfig instance
        """
        return cls(
            camera=CameraConfig.from_env(),
            streaming=StreamingConfig.from_env(),
            continuous_move=ContinuousMoveConfig.from_env(),
        )
