"""ONVIF Client - Worker Thread for Camera Control"""

import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Any

from jetson_zoom.config import CameraConfig, ContinuousMoveConfig
from jetson_zoom.logger import get_logger


class ZoomDirection(Enum):
    """Zoom direction enumeration."""

    IN = 1.0
    OUT = -1.0
    STOP = 0.0


@dataclass
class PTZMoveCommand:
    """ContinuousMove command for ONVIF PTZ."""

    pan_x: float = 0.0  # -1..1 (left/right)
    pan_y: float = 0.0  # -1..1 (down/up)
    zoom_x: float = 0.0  # -1..1 (out/in)
    duration_ms: int = 500  # How long to apply movement; <0 = hold until STOP


@dataclass
class PTZStopCommand:
    """Stop command for ONVIF PTZ."""

    pan_tilt: bool = True
    zoom: bool = True


class ONVIFClient(threading.Thread):
    """Worker thread: Executes ONVIF/SOAP commands for camera control.

    Architecture:
    - Runs in daemon thread pool
    - Handles continuous move logic (velocity-based control)
    - Implements safety: always sends STOP after movement
    - Non-blocking command queue processing

    Workflow:
    1. Receive move command from queue
    2. Send velocity command to camera (SOAP)
    3. Wait for specified interval
    4. Send STOP command
    5. Ready for next command
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        continuous_move_config: ContinuousMoveConfig,
        command_queue: queue.Queue,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize ONVIF client.

        Args:
            camera_config: Camera connection settings
            continuous_move_config: Zoom movement parameters
            command_queue: Queue to receive movement commands
            error_callback: Optional callback for error messages
        """
        super().__init__(name="ONVIFClient", daemon=True)

        self.logger = get_logger(self.__class__.__name__)
        self.camera_config = camera_config
        self.continuous_move_config = continuous_move_config
        self.command_queue = command_queue
        self.error_callback = error_callback

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._camera: Any = None
        self._media: Any = None
        self._ptz: Any = None
        self._profile_token: Optional[str] = None
        self._ptz_configuration_token: Optional[str] = None
        self._state_lock = threading.Lock()
        self._last_zoom_position: Optional[float] = None
        self._zoom_supported: Optional[bool] = None
        self._pan_tilt_supported: Optional[bool] = None
        self._last_error_lock = threading.Lock()
        self._last_error: Optional[str] = None

    def run(self) -> None:
        """Main thread execution: process ONVIF commands from queue."""
        try:
            self._connect_onvif()
            self._ready_event.set()
            self._set_last_error(None)
            self.logger.info("ONVIF client initialized (ready)")

            last_status_poll = 0.0
            poll_interval_s = 0.5

            while not self._stop_event.is_set():
                try:
                    # Non-blocking get with timeout
                    command = self.command_queue.get(timeout=0.2)
                    self._execute_command(command)
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_status_poll >= poll_interval_s:
                        last_status_poll = now
                        self._update_zoom_status()
                    continue
                except Exception as e:
                    error_msg = f"Command execution error: {e}"
                    self.logger.error(error_msg, exc_info=True)
                    self._set_last_error(error_msg)
                    if self.error_callback:
                        self.error_callback(error_msg)

        except Exception as e:
            error_msg = f"ONVIF client error: {e}"
            self.logger.error(error_msg, exc_info=True)
            self._set_last_error(str(e))
            if self.error_callback:
                self.error_callback(error_msg)
        finally:
            self._ready_event.clear()
            self._cleanup()

    def _connect_onvif(self) -> None:
        """Establish ONVIF connection and get device profile.

        Raises:
            RuntimeError: If connection or profile retrieval fails
        """
        try:
            ONVIFCamera = self._import_onvif_camera()
            transport = self._build_transport()

            self.logger.info(
                f"Connecting ONVIF: {self.camera_config.host}:{self.camera_config.port_onvif}"
            )

            try:
                self._camera = ONVIFCamera(
                    self.camera_config.host,
                    self.camera_config.port_onvif,
                    self.camera_config.username,
                    self.camera_config.password,
                    transport=transport,
                )
            except TypeError:
                # Some onvif-zeep versions may not accept `transport`.
                self._camera = ONVIFCamera(
                    self.camera_config.host,
                    self.camera_config.port_onvif,
                    self.camera_config.username,
                    self.camera_config.password,
                )

            self._media = self._camera.create_media_service()
            self._ptz = self._camera.create_ptz_service()

            self._profile_token, self._ptz_configuration_token = self._select_profile_tokens()
            if not self._profile_token:
                raise RuntimeError("No ONVIF media profiles found on camera")

            self.logger.info(f"ONVIF connected. ProfileToken={self._profile_token}")
            self._update_zoom_status()
            self._zoom_supported, self._pan_tilt_supported = self._detect_ptz_support()
            if self._zoom_supported is False:
                self.logger.warning(
                    "Camera does not advertise ONVIF PTZ Zoom control (may be fixed-lens/varifocal or digital-zoom-only)."
                )
            if self._pan_tilt_supported is False:
                self.logger.warning("Camera does not advertise ONVIF PTZ Pan/Tilt control.")

        except Exception as e:
            url = self.camera_config.build_onvif_url()
            raise RuntimeError(
                "ONVIF connection failed. "
                f"Endpoint={url}. "
                "Kiểm tra CAMERA_HOST/CAMERA_PORT_ONVIF, cùng mạng, và bật ONVIF trên camera. "
                f"Chi tiết: {e}"
            ) from e

    def _select_profile_tokens(self) -> tuple[Optional[str], Optional[str]]:
        """Choose a media profile token and (if available) PTZ configuration token."""
        try:
            if not self._media:
                return None, None

            profiles = self._media.GetProfiles()
            if not profiles:
                return None, None

            # Prefer a profile that actually has PTZConfiguration.
            for p in profiles:
                token = getattr(p, "token", None)
                ptz_cfg = getattr(p, "PTZConfiguration", None)
                ptz_cfg_token = getattr(ptz_cfg, "token", None) if ptz_cfg is not None else None
                if token and ptz_cfg is not None:
                    return token, ptz_cfg_token

            # Fallback: first media profile.
            return getattr(profiles[0], "token", None), None
        except Exception as e:
            self.logger.warning(f"Could not retrieve profile: {e}")
            return None, None

    def _execute_command(self, command: Any) -> None:
        """Execute a PTZ command with responsive preemption.

        Workflow:
        1. Send velocity command
        2. Wait for specified duration (interruptible)
        3. Send STOP command
        4. Log completion

        Args:
            command: PTZMoveCommand or PTZStopCommand
        """
        self.logger.debug(f"Executing move command: {command}")

        try:
            while True:
                if isinstance(command, PTZStopCommand):
                    self._send_stop(pan_tilt=command.pan_tilt, zoom=command.zoom)
                    self._update_zoom_status()
                    return

                if not isinstance(command, PTZMoveCommand):
                    raise RuntimeError(f"Unknown command type: {type(command)}")

                pan_x = max(-1.0, min(1.0, float(command.pan_x)))
                pan_y = max(-1.0, min(1.0, float(command.pan_y)))
                zoom_x = max(-1.0, min(1.0, float(command.zoom_x)))

                duration_ms = int(command.duration_ms)

                # Hold mode: duration_ms < 0 means "start continuous move and return"
                if duration_ms < 0:
                    # For "hold" mode we allow velocity updates without forcing a STOP first.
                    # This tends to feel smoother (especially for mouse-drag joystick control).
                    self._send_continuous_move(pan_x=pan_x, pan_y=pan_y, zoom_x=zoom_x)
                    return

                self._send_continuous_move(pan_x=pan_x, pan_y=pan_y, zoom_x=zoom_x)

                # Wait for specified interval (interruptible by newer command)
                if duration_ms > 0:
                    end_at = time.monotonic() + (duration_ms / 1000.0)
                    preempted = False
                    while time.monotonic() < end_at:
                        if self._stop_event.is_set():
                            break
                        try:
                            next_cmd = self.command_queue.get_nowait()
                        except queue.Empty:
                            time.sleep(0.02)
                            continue

                        # Preempt current move
                        try:
                            self._send_stop(pan_tilt=True, zoom=True)
                        except Exception:
                            pass
                        command = next_cmd
                        preempted = True
                        break

                    if preempted:
                        continue

                # Always send STOP to prevent motor runaway
                self._send_stop(pan_tilt=True, zoom=True)
                self._update_zoom_status()
                return

        except Exception as e:
            self.logger.error(f"Move command failed: {e}", exc_info=True)
            self._set_last_error(f"Move command failed: {e}")
            # Attempt emergency stop
            try:
                self._send_stop(pan_tilt=True, zoom=True)
                self._update_zoom_status()
            except Exception as stop_error:
                self.logger.error(f"Emergency stop failed: {stop_error}")
                self._set_last_error(f"Emergency stop failed: {stop_error}")

    def _send_continuous_move(self, pan_x: float = 0.0, pan_y: float = 0.0, zoom_x: float = 0.0) -> None:
        """Send continuous move command to camera via ONVIF.

        Args:
            pan_x: Pan velocity (-1.0..1.0)
            pan_y: Tilt velocity (-1.0..1.0)
            zoom_x: Zoom velocity (-1.0..1.0). Positive = zoom in.
        """
        if not self._ptz or not self._profile_token:
            raise RuntimeError("ONVIF PTZ service not initialized")

        self.logger.debug(
            f"Sending ContinuousMove pan=({pan_x:.2f},{pan_y:.2f}) zoom={zoom_x:.2f}"
        )

        # Build velocity payload. Some cameras reject PanTilt when Zoom-only is intended,
        # so we omit unused axes.
        velocity_payload: dict[str, Any] = {}
        if abs(pan_x) > 1e-6 or abs(pan_y) > 1e-6:
            velocity_payload["PanTilt"] = {"x": float(pan_x), "y": float(pan_y)}
        if abs(zoom_x) > 1e-6:
            velocity_payload["Zoom"] = {"x": float(zoom_x)}

        if not velocity_payload:
            # No movement requested.
            return

        try:
            request = self._ptz.create_type("ContinuousMove")
            request.ProfileToken = self._profile_token
            request.Velocity = velocity_payload
            self._ptz.ContinuousMove(request)
        except Exception:
            # Fallback shape for cameras that don't accept zeep objects.
            self._ptz.ContinuousMove(
                {
                    "ProfileToken": self._profile_token,
                    "Velocity": velocity_payload,
                }
            )

    def _send_stop(self, pan_tilt: bool = False, zoom: bool = True) -> None:
        """Send STOP command to halt movement.

        Critical for safety: prevents motor from overshooting
        """
        if not self._ptz or not self._profile_token:
            raise RuntimeError("ONVIF PTZ service not initialized")

        self.logger.debug(f"Sending PTZ Stop (PanTilt={pan_tilt}, Zoom={zoom})")

        try:
            request = self._ptz.create_type("Stop")
            request.ProfileToken = self._profile_token
            request.PanTilt = bool(pan_tilt)
            request.Zoom = bool(zoom)
            self._ptz.Stop(request)
        except Exception:
            self._ptz.Stop(
                {
                    "ProfileToken": self._profile_token,
                    "PanTilt": bool(pan_tilt),
                    "Zoom": bool(zoom),
                }
            )

    def _update_zoom_status(self) -> None:
        """Fetch and cache the camera's current optical zoom position (if supported).

        Many cameras expose PTZ status position including Zoom.x. If unsupported,
        this method silently does nothing.
        """
        if not self._ptz or not self._profile_token:
            return

        try:
            status = self._ptz.GetStatus({"ProfileToken": self._profile_token})
            position = getattr(status, "Position", None) if status is not None else None
            zoom = getattr(position, "Zoom", None) if position is not None else None
            x = getattr(zoom, "x", None) if zoom is not None else None
            if x is None:
                return
            with self._state_lock:
                self._last_zoom_position = float(x)
        except Exception:
            return

    def get_zoom_position(self) -> Optional[float]:
        """Return last known optical zoom position reported by the camera (Zoom.x)."""
        with self._state_lock:
            return self._last_zoom_position

    def is_zoom_supported(self) -> Optional[bool]:
        """Return whether camera advertises PTZ Zoom control via ONVIF.

        - True: camera reports zoom spaces/capabilities
        - False: camera reports PTZ but no zoom control spaces
        - None: unknown (probe failed / camera doesn't expose options)
        """
        return self._zoom_supported

    def is_pan_tilt_supported(self) -> Optional[bool]:
        """Return whether camera advertises PTZ Pan/Tilt control via ONVIF."""
        return self._pan_tilt_supported

    def queue_zoom_command(
        self,
        direction: ZoomDirection,
        velocity: Optional[float] = None,
        duration_ms: int = 500,
    ) -> bool:
        """Queue a zoom command for execution.

        Args:
            direction: Zoom direction (IN/OUT/STOP)
            velocity: Velocity (0.1-1.0). Uses config default if None
            duration_ms: How long to apply command

        Returns:
            True if command queued successfully, False if queue is full
        """
        if not self._ready_event.is_set():
            self.logger.warning("ONVIF chưa sẵn sàng (chưa kết nối), bỏ qua lệnh zoom")
            return False

        if velocity is None:
            velocity = self.continuous_move_config.zoom_velocity

        # Clamp velocity to valid range
        velocity = max(0.1, min(1.0, float(velocity)))

        if direction == ZoomDirection.STOP:
            command: Any = PTZStopCommand(pan_tilt=False, zoom=True)
        else:
            command = PTZMoveCommand(
                pan_x=0.0,
                pan_y=0.0,
                zoom_x=velocity * float(direction.value),
                duration_ms=int(duration_ms),
            )

        try:
            self._put_latest_command(command)
            self.logger.debug(f"Zoom command queued: {command}")
            return True
        except queue.Full:
            self.logger.warning("Command queue full; could not queue zoom command")
            return False

    def queue_pan_tilt_command(
        self,
        pan_x: float,
        pan_y: float,
        velocity: Optional[float] = None,
        duration_ms: int = 500,
        hold: bool = False,
    ) -> bool:
        """Queue a pan/tilt command for execution."""
        if not self._ready_event.is_set():
            self.logger.warning("ONVIF chưa sẵn sàng (chưa kết nối), bỏ qua lệnh pan/tilt")
            return False

        if velocity is None:
            # Use a single "PT velocity" knob for both axes (default = config pan_velocity)
            velocity = self.continuous_move_config.pan_velocity

        velocity = max(0.1, min(1.0, float(velocity)))
        pan_x = max(-1.0, min(1.0, float(pan_x))) * velocity
        pan_y = max(-1.0, min(1.0, float(pan_y))) * velocity

        cmd = PTZMoveCommand(
            pan_x=pan_x,
            pan_y=pan_y,
            zoom_x=0.0,
            duration_ms=-1 if hold else int(duration_ms),
        )

        try:
            self._put_latest_command(cmd)
            self.logger.debug(f"Pan/Tilt command queued: {cmd}")
            return True
        except queue.Full:
            self.logger.warning("Command queue full; could not queue pan/tilt command")
            return False

    def queue_stop(self, pan_tilt: bool = True, zoom: bool = True) -> bool:
        """Queue a STOP command."""
        if not self._ready_event.is_set():
            self.logger.warning("ONVIF chưa sẵn sàng (chưa kết nối), bỏ qua STOP")
            return False
        cmd = PTZStopCommand(pan_tilt=bool(pan_tilt), zoom=bool(zoom))
        try:
            self._put_latest_command(cmd)
            self.logger.debug(f"Stop command queued: {cmd}")
            return True
        except queue.Full:
            self.logger.warning("Command queue full; could not queue STOP")
            return False

    def _put_latest_command(self, command: Any) -> None:
        """Keep UI responsive: drop queued commands and keep only the newest."""
        try:
            while True:
                self.command_queue.get_nowait()
        except queue.Empty:
            pass
        self.command_queue.put_nowait(command)

    def stop(self) -> None:
        """Stop the client gracefully."""
        self.logger.info("Stopping ONVIF client...")
        self._stop_event.set()

    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def get_last_error(self) -> Optional[str]:
        with self._last_error_lock:
            return self._last_error

    def _set_last_error(self, message: Optional[str]) -> None:
        with self._last_error_lock:
            self._last_error = message

    def _cleanup(self) -> None:
        """Clean up ONVIF resources."""
        try:
            if self._ptz and self._profile_token:
                self._send_stop(pan_tilt=True, zoom=True)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def _detect_ptz_support(self) -> tuple[Optional[bool], Optional[bool]]:
        """Best-effort probe for whether the camera supports PTZ Zoom and Pan/Tilt.

        Many consumer cameras expose ONVIF but do not expose motorized zoom. In that case,
        ContinuousMove(Zoom) is ignored or returns a SOAP fault (ActionNotSupported).

        Returns:
            (zoom_supported, pan_tilt_supported) where each is True/False/None.
        """
        if not self._ptz:
            return None, None

        token = self._ptz_configuration_token
        if not token:
            # Try fallback: use first configuration token if available.
            try:
                configs = self._ptz.GetConfigurations()
                if configs:
                    token = getattr(configs[0], "token", None) or getattr(configs[0], "_token", None)
            except Exception:
                token = None

        if not token:
            return None, None

        try:
            options = self._ptz.GetConfigurationOptions({"ConfigurationToken": token})
        except Exception:
            return None, None

        # Support both zeep objects and dict-like payloads.
        spaces = getattr(options, "Spaces", None)
        if spaces is None and isinstance(options, dict):
            spaces = options.get("Spaces")

        if spaces is None:
            return None, None

        zoom_spaces = getattr(spaces, "ContinuousZoomVelocitySpace", None)
        if zoom_spaces is None and isinstance(spaces, dict):
            zoom_spaces = spaces.get("ContinuousZoomVelocitySpace")

        pan_tilt_spaces = getattr(spaces, "ContinuousPanTiltVelocitySpace", None)
        if pan_tilt_spaces is None and isinstance(spaces, dict):
            pan_tilt_spaces = spaces.get("ContinuousPanTiltVelocitySpace")

        zoom_supported: Optional[bool]
        pan_tilt_supported: Optional[bool]

        if zoom_spaces is None:
            zoom_supported = False
        else:
            try:
                zoom_supported = len(list(zoom_spaces)) > 0
            except Exception:
                zoom_supported = True

        if pan_tilt_spaces is None:
            pan_tilt_supported = False
        else:
            try:
                pan_tilt_supported = len(list(pan_tilt_spaces)) > 0
            except Exception:
                pan_tilt_supported = True

        return zoom_supported, pan_tilt_supported

    @staticmethod
    def _import_onvif_camera():
        try:
            from onvif import ONVIFCamera  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ONVIF library not found. Install with: pip install onvif-zeep"
            ) from e
        return ONVIFCamera

    def _build_transport(self):
        try:
            from zeep.transports import Transport  # type: ignore
        except Exception:  # pragma: no cover
            return None

        timeout = getattr(self.camera_config, "onvif_timeout", None)
        try:
            return Transport(timeout=timeout)
        except Exception:
            return None
