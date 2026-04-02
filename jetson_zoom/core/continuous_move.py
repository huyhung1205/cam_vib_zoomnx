"""Continuous Move Logic for Smooth Camera Control

Implements velocity-based zoom control with safety mechanisms:
- Velocity-controlled movement (0.1 - 1.0)
- Fixed interval-based application (e.g., 500ms)
- Mandatory STOP command after each move
- Queue-based command submission
"""

from typing import Optional
import logging

from jetson_zoom.config import ContinuousMoveConfig
from jetson_zoom.controllers.onvif_client import ONVIFClient, ZoomDirection
from jetson_zoom.logger import get_logger


class ContinuousMover:
    """Manages continuous zoom movement with safety mechanisms.

    Design Principles:
    1. Velocity-based control (smooth movement)
    2. Time-interval execution (e.g., 500ms bursts)
    3. Always STOP after movement (safety)
    4. Non-blocking command submission
    """

    def __init__(
        self,
        onvif_client: ONVIFClient,
        config: ContinuousMoveConfig,
    ) -> None:
        """Initialize continuous mover.

        Args:
            onvif_client: ONVIF client for sending commands
            config: Continuous move configuration
        """
        self.logger = get_logger(self.__class__.__name__)
        self.onvif_client = onvif_client
        self.config = config

        # Cached value reported by camera (PTZ GetStatus Position.Zoom.x), if available.
        self._current_zoom_level: Optional[float] = None

    def zoom_in(
        self,
        velocity: Optional[float] = None,
        duration_ms: Optional[int] = None,
    ) -> bool:
        """Queue zoom-in command.

        Args:
            velocity: Zoom velocity (0.1-1.0). Uses config default if None.
            duration_ms: Command duration. Uses config default if None.

        Returns:
            True if command queued successfully
        """
        velocity = velocity or self.config.zoom_velocity
        duration_ms = duration_ms or self.config.move_interval_ms

        self.logger.info(f"Zoom IN: velocity={velocity}, duration={duration_ms}ms")

        success = self.onvif_client.queue_zoom_command(
            direction=ZoomDirection.IN,
            velocity=velocity,
            duration_ms=duration_ms,
        )

        if not success:
            self.logger.warning("Zoom IN: gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    def zoom_in_hold(self, velocity: Optional[float] = None) -> bool:
        """Start continuous (hold) zoom-in. Call `stop_movement()` on release."""
        velocity = velocity or self.config.zoom_velocity
        self.logger.info(f"Zoom IN (HOLD): velocity={velocity}")
        success = self.onvif_client.queue_zoom_command(
            direction=ZoomDirection.IN,
            velocity=velocity,
            duration_ms=-1,
        )
        if not success:
            self.logger.warning("Zoom IN (HOLD): gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    def zoom_out(
        self,
        velocity: Optional[float] = None,
        duration_ms: Optional[int] = None,
    ) -> bool:
        """Queue zoom-out command.

        Args:
            velocity: Zoom velocity (0.1-1.0). Uses config default if None.
            duration_ms: Command duration. Uses config default if None.

        Returns:
            True if command queued successfully
        """
        velocity = velocity or self.config.zoom_velocity
        duration_ms = duration_ms or self.config.move_interval_ms

        self.logger.info(f"Zoom OUT: velocity={velocity}, duration={duration_ms}ms")

        success = self.onvif_client.queue_zoom_command(
            direction=ZoomDirection.OUT,
            velocity=velocity,
            duration_ms=duration_ms,
        )

        if not success:
            self.logger.warning("Zoom OUT: gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    def zoom_out_hold(self, velocity: Optional[float] = None) -> bool:
        """Start continuous (hold) zoom-out. Call `stop_movement()` on release."""
        velocity = velocity or self.config.zoom_velocity
        self.logger.info(f"Zoom OUT (HOLD): velocity={velocity}")
        success = self.onvif_client.queue_zoom_command(
            direction=ZoomDirection.OUT,
            velocity=velocity,
            duration_ms=-1,
        )
        if not success:
            self.logger.warning("Zoom OUT (HOLD): gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    def stop_movement(self) -> bool:
        """Queue STOP for zoom axis only.

        Returns:
            True if STOP command queued successfully
        """
        self.logger.info("Movement STOP requested")

        success = self.onvif_client.queue_stop(pan_tilt=False, zoom=True)
        if not success:
            self.logger.warning("STOP: gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    def stop_pan_tilt(self) -> bool:
        """Queue STOP for pan/tilt axis only."""
        self.logger.info("Pan/Tilt STOP requested")
        success = self.onvif_client.queue_stop(pan_tilt=True, zoom=False)
        if not success:
            self.logger.warning("STOP(Pan/Tilt): gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    def stop_all(self) -> bool:
        """Queue STOP for all axes."""
        self.logger.info("STOP ALL requested")
        success = self.onvif_client.queue_stop(pan_tilt=True, zoom=True)
        if not success:
            self.logger.warning("STOP(ALL): gửi lệnh thất bại (ONVIF chưa sẵn sàng hoặc bị thay thế)")
        return success

    # Pan/Tilt --------------------------------------------------------------

    def pan_left(self, velocity: Optional[float] = None, duration_ms: Optional[int] = None) -> bool:
        velocity = velocity or self.config.pan_velocity
        duration_ms = duration_ms or self.config.move_interval_ms
        return self.onvif_client.queue_pan_tilt_command(
            pan_x=-1.0, pan_y=0.0, velocity=velocity, duration_ms=int(duration_ms), hold=False
        )

    def pan_right(self, velocity: Optional[float] = None, duration_ms: Optional[int] = None) -> bool:
        velocity = velocity or self.config.pan_velocity
        duration_ms = duration_ms or self.config.move_interval_ms
        return self.onvif_client.queue_pan_tilt_command(
            pan_x=1.0, pan_y=0.0, velocity=velocity, duration_ms=int(duration_ms), hold=False
        )

    def tilt_up(self, velocity: Optional[float] = None, duration_ms: Optional[int] = None) -> bool:
        velocity = velocity or self.config.tilt_velocity
        duration_ms = duration_ms or self.config.move_interval_ms
        return self.onvif_client.queue_pan_tilt_command(
            pan_x=0.0, pan_y=1.0, velocity=velocity, duration_ms=int(duration_ms), hold=False
        )

    def tilt_down(self, velocity: Optional[float] = None, duration_ms: Optional[int] = None) -> bool:
        velocity = velocity or self.config.tilt_velocity
        duration_ms = duration_ms or self.config.move_interval_ms
        return self.onvif_client.queue_pan_tilt_command(
            pan_x=0.0, pan_y=-1.0, velocity=velocity, duration_ms=int(duration_ms), hold=False
        )

    def pan_left_hold(self, velocity: Optional[float] = None) -> bool:
        velocity = velocity or self.config.pan_velocity
        return self.onvif_client.queue_pan_tilt_command(pan_x=-1.0, pan_y=0.0, velocity=velocity, hold=True)

    def pan_right_hold(self, velocity: Optional[float] = None) -> bool:
        velocity = velocity or self.config.pan_velocity
        return self.onvif_client.queue_pan_tilt_command(pan_x=1.0, pan_y=0.0, velocity=velocity, hold=True)

    def tilt_up_hold(self, velocity: Optional[float] = None) -> bool:
        velocity = velocity or self.config.tilt_velocity
        return self.onvif_client.queue_pan_tilt_command(pan_x=0.0, pan_y=1.0, velocity=velocity, hold=True)

    def tilt_down_hold(self, velocity: Optional[float] = None) -> bool:
        velocity = velocity or self.config.tilt_velocity
        return self.onvif_client.queue_pan_tilt_command(pan_x=0.0, pan_y=-1.0, velocity=velocity, hold=True)

    def get_zoom_level(self) -> Optional[float]:
        """Get current zoom position.

        If camera supports ONVIF PTZ GetStatus, this returns the cached optical zoom
        position `Zoom.x`. If unsupported/unavailable, returns None.

        Returns:
            Current zoom level
        """
        try:
            value = self.onvif_client.get_zoom_position()
            if value is not None:
                self._current_zoom_level = float(value)
        except Exception:
            pass
        return self._current_zoom_level

    def set_zoom_level(self, level: float) -> None:
        """Set zoom level (for testing/overrides).

        Args:
            level: Target zoom level (clamped to min/max)
        """
        self._current_zoom_level = float(
            max(
                self.config.zoom_min,
                min(self.config.zoom_max, level),
            )
        )
        self.logger.debug(f"Zoom level set to: {self._current_zoom_level:.4f}")
