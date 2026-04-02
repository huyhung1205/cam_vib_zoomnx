"""Event Loop - Main Thread for Display and Event Handling

Responsibilities:
1. Display video frames from RTSP queue
2. Handle keyboard/mouse input for zoom control
3. Monitor and log application status
4. Manage thread lifecycle
"""

import queue
import threading
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from jetson_zoom.logger import get_logger
from jetson_zoom.config import ApplicationConfig
from jetson_zoom.streams.rtsp_handler import VideoFrame, RTSPStreamHandler
from jetson_zoom.core.continuous_move import ContinuousMover


class EventHandler(ABC):
    """Abstract base for event handling."""

    @abstractmethod
    def on_key_press(self, key: str) -> None:
        """Handle keyboard input.

        Args:
            key: Key code or character
        """
        pass

    @abstractmethod
    def on_frame_received(self, frame: VideoFrame) -> None:
        """Handle new video frame.

        Args:
            frame: VideoFrame object
        """
        pass


class EventLoop:
    """Main thread: Manages display and event processing.

    Architecture:
    - Pulls frames from RTSP producer queue
    - Handles user input (keyboard for zoom control)
    - Maintains display at target FPS
    - Monitors and logs application metrics

    Workflow:
    1. Initialize all threads
    2. Pull frame from queue (or skip if empty)
    3. Process input events
    4. Display frame
    5. Log metrics
    6. Repeat until shutdown
    """

    def __init__(
        self,
        config: ApplicationConfig,
        continuous_mover: ContinuousMover,
        rtsp_handler: RTSPStreamHandler,
        event_handler: Optional[EventHandler] = None,
    ) -> None:
        """Initialize event loop.

        Args:
            config: Application configuration
            continuous_mover: ContinuousMover instance for zoom control
            rtsp_handler: RTSP stream handler thread
            event_handler: Optional custom event handler
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.continuous_mover = continuous_mover
        self.rtsp_handler = rtsp_handler
        self.event_handler = event_handler

        self._running = False
        self._stop_event = threading.Event()

        # Metrics
        self._frames_displayed = 0
        self._frames_dropped = 0
        self._start_time = time.time()

        # Target frame timing
        self._frame_time_ms = 1000.0 / config.streaming.target_fps

    def run(self) -> None:
        """Main event loop: Process frames and handle events.

        This is the main thread function that should run on the main thread
        due to GUI/display requirements.
        """
        self.logger.info("Event loop starting...")
        self._running = True
        self._start_time = time.time()

        display_backend = (self.config.streaming.display_backend or "opencv").strip().lower()
        cv2 = None
        if display_backend == "opencv":
            cv2 = self._import_cv2()
            cv2.namedWindow(self.config.streaming.window_name, cv2.WINDOW_NORMAL)

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()

                # Try to get next frame
                frame = self._process_frame()
                if frame is not None and cv2 is not None:
                    self._display_frame(cv2, frame)

                # Handle input events (non-blocking)
                if cv2 is not None:
                    self._process_input_opencv(cv2)

                # Monitor metrics periodically
                self._check_metrics()

                # Frame rate regulation
                elapsed_ms = (time.time() - loop_start) * 1000.0
                sleep_ms = self._frame_time_ms - elapsed_ms
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Event loop error: {e}", exc_info=True)
        finally:
            self._cleanup()
            if cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def _process_frame(self) -> Optional[VideoFrame]:
        """Get the most recent frame from RTSP queue (drain-oldest to reduce latency)."""
        try:
            # Grab one frame, then drain the queue to keep the newest (lower latency).
            frame: VideoFrame = self.rtsp_handler.output_queue.get(timeout=0.01)
            drained = 0
            while True:
                try:
                    frame = self.rtsp_handler.output_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break

            if drained:
                self._frames_dropped += drained
            self._frames_displayed += 1

            if self._frames_displayed % 100 == 0:
                self.logger.debug(f"Displayed {self._frames_displayed} frames")

            if self.event_handler is not None:
                try:
                    self.event_handler.on_frame_received(frame)
                except Exception:
                    pass
            return frame

        except queue.Empty:
            # No frame available - skip this iteration
            self._frames_dropped += 1
            return None

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return None

    def _process_input_opencv(self, cv2) -> None:
        """Read keyboard input via OpenCV window and route to `handle_key_press`."""
        try:
            if cv2.getWindowProperty(self.config.streaming.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.stop()
                return
        except Exception:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return

        try:
            self.handle_key_press(chr(key))
        except Exception:
            # Ignore non-ASCII keys
            return

    def _display_frame(self, cv2, frame: VideoFrame) -> None:
        """Render a frame using OpenCV."""
        image = frame.image

        try:
            if (
                self.config.streaming.display_width > 0
                and self.config.streaming.display_height > 0
            ):
                image = cv2.resize(
                    image,
                    (self.config.streaming.display_width, self.config.streaming.display_height),
                    interpolation=cv2.INTER_AREA,
                )
        except Exception:
            pass

        # Overlay a small status line (optional)
        try:
            status = self.get_status()
            zoom_level = status.get("zoom_level")
            zoom_text = f"{float(zoom_level):.2f}" if zoom_level is not None else "n/a"
            overlay = (
                f"FPS {status['actual_fps']:.1f}/{status['target_fps']} | "
                f"Dropped {status['frames_dropped']} | "
                f"ONVIF {'OK' if status.get('onvif_ready') else 'NO'} | "
                f"Zoom.x {zoom_text}"
            )
            cv2.putText(
                image,
                overlay,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

        cv2.imshow(self.config.streaming.window_name, image)

    def handle_key_press(self, key: str) -> None:
        """Public method to handle keyboard input (called by GUI integration).

        Args:
            key: Key code or character
        """
        key_lower = key.lower().strip()

        if self.event_handler is not None:
            try:
                self.event_handler.on_key_press(key_lower)
            except Exception:
                pass

        if key_lower == "i":
            self.continuous_mover.zoom_in()
        elif key_lower == "o":
            self.continuous_mover.zoom_out()
        elif key_lower == "s":
            self.continuous_mover.stop_movement()
        elif key_lower == "q":
            self.stop()
        else:
            self.logger.debug(f"Unknown key: {key}")

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OpenCV (cv2) is required for DISPLAY_BACKEND=opencv. "
                "On Windows: pip install opencv-python. "
                "On Jetson: sudo apt-get install python3-opencv."
            ) from e
        return cv2

    def _check_metrics(self) -> None:
        """Periodically report performance metrics."""
        if self._frames_displayed % 300 == 0:  # Every 300 frames
            elapsed = time.time() - self._start_time
            actual_fps = self._frames_displayed / elapsed if elapsed > 0 else 0
            drop_rate = (
                self._frames_dropped
                / (self._frames_displayed + self._frames_dropped)
                * 100.0
                if (self._frames_displayed + self._frames_dropped) > 0
                else 0
            )

            self.logger.info(
                f"Metrics - FPS: {actual_fps:.1f}, "
                f"Displayed: {self._frames_displayed}, "
                f"Dropped: {self._frames_dropped} ({drop_rate:.1f}%), "
                f"Elapsed: {elapsed:.1f}s"
            )

            # Log RTSP handler stats
            try:
                stats = self.rtsp_handler.get_stats()
                self.logger.debug(f"RTSP stats: {stats}")
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the event loop gracefully."""
        self.logger.info("Stopping event loop...")
        self._stop_event.set()

    def _cleanup(self) -> None:
        """Clean up and shutdown all threads."""
        self._running = False

        self.logger.info("Shutting down threads...")

        # Stop RTSP handler
        try:
            self.rtsp_handler.stop()
            self.rtsp_handler.join(timeout=5.0)
        except Exception as e:
            self.logger.warning(f"Error stopping RTSP handler: {e}")

        # Stop ONVIF client
        try:
            self.continuous_mover.onvif_client.stop()
            self.continuous_mover.onvif_client.join(timeout=5.0)
        except Exception as e:
            self.logger.warning(f"Error stopping ONVIF client: {e}")

        # Final metrics
        elapsed = time.time() - self._start_time
        self.logger.info(
            f"Application shutdown - Total frames: {self._frames_displayed}, "
            f"Total dropped: {self._frames_dropped}, "
            f"Duration: {elapsed:.1f}s"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current application status.

        Returns:
            Dictionary with status information
        """
        elapsed = time.time() - self._start_time
        actual_fps = self._frames_displayed / elapsed if elapsed > 0 else 0

        return {
            "running": self._running,
            "frames_displayed": self._frames_displayed,
            "frames_dropped": self._frames_dropped,
            "actual_fps": actual_fps,
            "target_fps": self.config.streaming.target_fps,
            "zoom_level": self.continuous_mover.get_zoom_level(),
            "onvif_ready": self.continuous_mover.onvif_client.is_ready(),
            "elapsed_seconds": elapsed,
        }
