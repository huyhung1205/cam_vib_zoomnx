"""JetsonZoom - Realtime Camera Control Application
Dual-Stack Architecture: RTSP for streaming, ONVIF for control
"""

__version__ = "1.0.0"
__author__ = "JetsonZoom Team"

from jetson_zoom.logger import get_logger

__all__ = ["get_logger"]
