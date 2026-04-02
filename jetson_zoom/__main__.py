"""JetsonZoom - Main Entry Point (Operational Workflow)

This module implements the complete operational workflow as described in README.md:

Step 1: Load configuration
Step 2: Start RTSP producer thread
Step 3: Start ONVIF worker thread
Step 4: Create continuous move controller
Step 5: Run main (display/input) event loop
Step 6: Cleanup on shutdown
"""

import sys
import queue
import signal
import argparse
import os
from urllib.parse import urlparse
from pathlib import Path

from jetson_zoom.logger import get_logger
from jetson_zoom.config import ApplicationConfig
from jetson_zoom.streams.rtsp_handler import RTSPStreamHandler
from jetson_zoom.controllers.onvif_client import ONVIFClient
from jetson_zoom.core.continuous_move import ContinuousMover
from jetson_zoom.core.event_loop import EventLoop
from jetson_zoom.sources import choose_source_interactive, find_source, load_sources
from jetson_zoom.ui.source_picker import pick_source_opencv
from jetson_zoom.state import load_state, state_path_from_env


def create_application(config: ApplicationConfig):
    """Factory function to create and wire up all application components.

    Args:
        config: ApplicationConfig instance

    Returns:
        Tuple of (rtsp_handler, onvif_client, continuous_mover, event_loop)
    """
    logger = get_logger("JetsonZoom")

    # Step 1: Initialize frame queue (RTSP producer -> main thread)
    frame_queue = queue.Queue(maxsize=max(1, config.streaming.frame_queue_size))

    # Step 2: Create RTSP stream handler (Producer thread)
    logger.info("Step 2/6: Initializing RTSP stream...")
    rtsp_handler = RTSPStreamHandler(
        camera_config=config.camera,
        streaming_config=config.streaming,
        output_queue=frame_queue,
    )
    rtsp_handler.start()

    # Step 3: Initialize command queue (main thread -> ONVIF worker)
    command_queue = queue.Queue(maxsize=10)

    # Step 4: Create ONVIF client (Worker thread)
    logger.info("Step 3/6: Initializing ONVIF controller...")
    onvif_client = ONVIFClient(
        camera_config=config.camera,
        continuous_move_config=config.continuous_move,
        command_queue=command_queue,
    )
    onvif_client.start()

    # Step 5: Create continuous move handler
    logger.info("Step 4/6: Setting up continuous zoom control...")
    continuous_mover = ContinuousMover(
        onvif_client=onvif_client,
        config=config.continuous_move,
    )

    # Step 6: Create event loop (Main thread)
    logger.info("Step 5/6: Creating event loop...")
    event_loop = EventLoop(
        config=config,
        continuous_mover=continuous_mover,
        rtsp_handler=rtsp_handler,
    )

    return rtsp_handler, onvif_client, continuous_mover, event_loop


def _default_sources_file() -> Path:
    # Repo root when running from source: JetsonZoom/jetson_zoom/__main__.py -> JetsonZoom/
    return Path(__file__).resolve().parent.parent / "sources.txt"


def _apply_rtsp_to_config(config: ApplicationConfig, rtsp_url: str) -> None:
    config.camera.rtsp_url = rtsp_url

    # Best-effort inference for CAMERA_HOST/CAMERA_PORT_RTSP if user didn't override them.
    parsed = urlparse(rtsp_url)
    if parsed.hostname:
        # Most cameras expose RTSP and ONVIF on the same host. Sync host to RTSP URL.
        config.camera.host = parsed.hostname
        if parsed.port:
            config.camera.port_rtsp = parsed.port


def _resolve_source(config: ApplicationConfig, args: argparse.Namespace) -> None:
    picker_mode = (os.getenv("SOURCE_PICKER", "auto") or "auto").strip().lower()
    force_picker = bool(getattr(args, "picker", False)) or picker_mode in {"always", "1", "true", "yes", "on"}

    # Explicit RTSP always wins
    if args.rtsp:
        _apply_rtsp_to_config(config, args.rtsp)
        return

    # If already configured via env (.env), don't prompt unless forced.
    if config.camera.rtsp_url and not force_picker:
        return

    sources_file = Path(args.sources_file) if args.sources_file else _default_sources_file()

    # Name-based selection (non-interactive)
    if args.source:
        sources = load_sources(sources_file)
        found = find_source(sources, args.source)
        if not found:
            raise RuntimeError(f"Không tìm thấy source '{args.source}' trong {sources_file}")
        _apply_rtsp_to_config(config, found.rtsp_url)
        return

    # UI selection:
    # - If forced: always use OpenCV picker (works in IDE).
    # - Otherwise: prefer terminal menu when possible.
    if not force_picker and sys.stdin is not None and sys.stdin.isatty():
        chosen, _ = choose_source_interactive(sources_file)
        if chosen is None:
            return
        _apply_rtsp_to_config(config, chosen.rtsp_url)
        return

    chosen = pick_source_opencv(sources_file)
    if chosen is None:
        return
    _apply_rtsp_to_config(config, chosen.rtsp_url)


def main() -> int:
    """Main entry point for JetsonZoom application.

    Workflow:
    1. Load configuration
    2. Create application components
    3. Run event loop
    4. Cleanup on exit

    Returns:
        Exit code (0 = success)
    """
    logger = get_logger("JetsonZoom.main")

    try:
        logger.info("=" * 60)
        logger.info("JetsonZoom v1.0.0 - Realtime Camera Control")
        logger.info("=" * 60)

        parser = argparse.ArgumentParser(prog="jetson-zoom")
        parser.add_argument("--sources-file", help="Path tới sources.txt (name|rtsp_url)")
        parser.add_argument("--source", help="Chọn source theo name trong sources file")
        parser.add_argument("--rtsp", help="RTSP URL (bỏ qua sources file)")
        parser.add_argument(
            "--ui",
            choices=["qt", "opencv"],
            default=os.getenv("UI", "qt").strip().lower() if os.getenv("UI") else "qt",
            help="Chọn giao diện: qt (khuyến nghị) hoặc opencv",
        )
        parser.add_argument("--state-file", help="Path tới file state.json (lưu cấu hình gần nhất)")
        parser.add_argument(
            "--picker",
            action="store_true",
            help="Luôn mở giao diện nhập/chọn RTSP (OpenCV Source Picker)",
        )
        args = parser.parse_args()

        # Load configuration from environment
        logger.info("Loading configuration...")
        config = ApplicationConfig.from_env()

        # Merge persisted state (last-used config) on top of env defaults
        state_path = Path(args.state_file) if args.state_file else state_path_from_env()
        state = load_state(state_path)
        if state:
            if state.host:
                config.camera.host = state.host
            if state.onvif_port:
                config.camera.port_onvif = int(state.onvif_port)
            if state.username:
                config.camera.username = state.username
            if state.password:
                config.camera.password = state.password
            if state.rtsp_url:
                config.camera.rtsp_url = state.rtsp_url

        sources_file = Path(args.sources_file) if args.sources_file else _default_sources_file()

        if args.ui == "qt":
            # Qt UI has its own in-app RTSP input; don't force CLI source resolution.
            try:
                from jetson_zoom.ui.qt_app import run_qt_ui
                return run_qt_ui(sources_file, config)
            except ImportError as e:
                logger.warning(f"Qt UI không khả dụng ({e}). Fallback sang UI OpenCV.")
                args.ui = "opencv"

        # OpenCV UI: Resolve RTSP source (sources.txt hoặc nhập mới)
        _resolve_source(config, args)

        # Create application components
        logger.info("Step 1/6: Wiring application components...")
        rtsp_handler, onvif_client, continuous_mover, event_loop = create_application(
            config
        )

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            event_loop.stop()

        try:
            signal.signal(signal.SIGINT, signal_handler)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception:
            pass

        # Run main event loop
        logger.info("Starting event loop (OpenCV window)...")
        event_loop.run()

        logger.info("Step 6/6: Application shutdown complete")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
