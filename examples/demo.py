"""Demo: How to use JetsonZoom programmatically.

This example shows how to:
1. Configure camera settings
2. Create application components
3. Handle zoom commands
4. Integrate with GUI frameworks
"""

import queue
import time
from jetson_zoom.config import (
    CameraConfig,
    StreamingConfig,
    ContinuousMoveConfig,
    ApplicationConfig,
)
from jetson_zoom.streams.rtsp_handler import RTSPStreamHandler
from jetson_zoom.controllers.onvif_client import ONVIFClient, ZoomDirection
from jetson_zoom.core.continuous_move import ContinuousMover
from jetson_zoom.core.event_loop import EventLoop
from jetson_zoom.logger import get_logger


def example_1_basic_setup():
    """Example 1: Basic setup and configuration."""
    logger = get_logger("example_1")
    logger.info("Example 1: Basic Setup")

    # Create configuration
    config = ApplicationConfig(
        camera=CameraConfig(
            host="192.168.1.70",
            port_rtsp=554,
            port_onvif=80,
            username="admin",
            password="12345",
        ),
        streaming=StreamingConfig(
            target_fps=30,
            display_width=1920,
            display_height=1080,
        ),
        continuous_move=ContinuousMoveConfig(
            zoom_velocity=0.5,
            move_interval_ms=500,
        ),
    )

    logger.info(f"Camera: {config.camera.host}:{config.camera.port_rtsp}")
    logger.info(f"Target FPS: {config.streaming.target_fps}")
    logger.info(f"Zoom velocity: {config.continuous_move.zoom_velocity}")


def example_2_thread_synchronization():
    """Example 2: Understanding thread synchronization with queues."""
    logger = get_logger("example_2")
    logger.info("Example 2: Thread Synchronization")

    config = ApplicationConfig.from_env()

    # Frame queue: RTSP Producer -> Main thread
    frame_queue = queue.Queue(maxsize=30)

    # Command queue: Main thread -> ONVIF Worker
    command_queue = queue.Queue(maxsize=10)

    logger.info(f"Frame queue max size: {frame_queue.maxsize}")
    logger.info(f"Command queue max size: {command_queue.maxsize}")

    # Demonstrate queue operations
    logger.info("Simulating queue push...")
    try:
        frame_queue.put_nowait("dummy_frame")
        logger.info(f"Frame queue size: {frame_queue.qsize()}")
    except queue.Full:
        logger.warning("Frame queue is full!")


def example_3_continuous_zoom():
    """Example 3: Using continuous zoom control."""
    logger = get_logger("example_3")
    logger.info("Example 3: Continuous Zoom Control")

    config = ApplicationConfig.from_env()

    # Create mock ONVIF client (would connect in real scenario)
    command_queue = queue.Queue(maxsize=10)
    onvif_client = ONVIFClient(
        camera_config=config.camera,
        continuous_move_config=config.continuous_move,
        command_queue=command_queue,
    )

    # Create continuous mover
    mover = ContinuousMover(
        onvif_client=onvif_client,
        config=config.continuous_move,
    )

    logger.info(f"Initial zoom level: {mover.get_zoom_level()}")

    # Simulate zoom commands
    logger.info("Simulating zoom IN...")
    # In real app: mover.zoom_in()  (would process in worker thread)

    logger.info("Simulating zoom OUT...")
    # In real app: mover.zoom_out()

    logger.info(f"Final zoom level: {mover.get_zoom_level()}")


def example_4_error_handling():
    """Example 4: Error handling and error callbacks."""
    logger = get_logger("example_4")
    logger.info("Example 4: Error Handling")

    def on_rtsp_error(error_msg: str):
        """Callback for RTSP errors."""
        logger.error(f"RTSP Error: {error_msg}")

    def on_onvif_error(error_msg: str):
        """Callback for ONVIF errors."""
        logger.error(f"ONVIF Error: {error_msg}")

    config = ApplicationConfig.from_env()
    frame_queue = queue.Queue()
    command_queue = queue.Queue()

    # Create handlers with error callbacks
    rtsp_handler = RTSPStreamHandler(
        camera_config=config.camera,
        streaming_config=config.streaming,
        output_queue=frame_queue,
        error_callback=on_rtsp_error,
    )

    onvif_client = ONVIFClient(
        camera_config=config.camera,
        continuous_move_config=config.continuous_move,
        command_queue=command_queue,
        error_callback=on_onvif_error,
    )

    logger.info("Error callbacks registered")


def example_5_metrics_monitoring():
    """Example 5: Monitoring application metrics."""
    logger = get_logger("example_5")
    logger.info("Example 5: Metrics Monitoring")

    config = ApplicationConfig.from_env()
    frame_queue = queue.Queue(maxsize=30)
    command_queue = queue.Queue(maxsize=10)

    rtsp_handler = RTSPStreamHandler(
        camera_config=config.camera,
        streaming_config=config.streaming,
        output_queue=frame_queue,
    )

    onvif_client = ONVIFClient(
        camera_config=config.camera,
        continuous_move_config=config.continuous_move,
        command_queue=command_queue,
    )

    mover = ContinuousMover(
        onvif_client=onvif_client,
        config=config.continuous_move,
    )

    event_loop = EventLoop(
        config=config,
        continuous_mover=mover,
        rtsp_handler=rtsp_handler,
    )

    # Get status (in real app, called periodically)
    status = event_loop.get_status()
    logger.info(f"Status: {status}")


if __name__ == "__main__":
    logger = get_logger("main")
    logger.info("JetsonZoom Examples")
    logger.info("=" * 60)

    try:
        example_1_basic_setup()
        logger.info("")

        example_2_thread_synchronization()
        logger.info("")

        example_3_continuous_zoom()
        logger.info("")

        example_4_error_handling()
        logger.info("")

        example_5_metrics_monitoring()

    except Exception as e:
        logger.error(f"Example error: {e}", exc_info=True)
