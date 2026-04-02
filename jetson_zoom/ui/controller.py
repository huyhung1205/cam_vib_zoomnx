from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Optional

from jetson_zoom.config import ApplicationConfig
from jetson_zoom.controllers.onvif_client import ONVIFClient
from jetson_zoom.core.continuous_move import ContinuousMover
from jetson_zoom.streams.rtsp_handler import RTSPStreamHandler, VideoFrame


@dataclass
class RunningApp:
    config: ApplicationConfig
    frame_queue: queue.Queue
    rtsp: RTSPStreamHandler
    onvif: ONVIFClient
    mover: ContinuousMover


class AppController:
    def __init__(self) -> None:
        self._running: Optional[RunningApp] = None

    @property
    def running(self) -> Optional[RunningApp]:
        return self._running

    def start(self, config: ApplicationConfig) -> RunningApp:
        self.stop()

        frame_queue: queue.Queue = queue.Queue(maxsize=max(1, config.streaming.frame_queue_size))
        rtsp = RTSPStreamHandler(
            camera_config=config.camera,
            streaming_config=config.streaming,
            output_queue=frame_queue,
        )
        rtsp.start()

        command_queue: queue.Queue = queue.Queue(maxsize=10)
        onvif = ONVIFClient(
            camera_config=config.camera,
            continuous_move_config=config.continuous_move,
            command_queue=command_queue,
        )
        onvif.start()

        mover = ContinuousMover(onvif_client=onvif, config=config.continuous_move)

        self._running = RunningApp(
            config=config,
            frame_queue=frame_queue,
            rtsp=rtsp,
            onvif=onvif,
            mover=mover,
        )
        return self._running

    def stop(self) -> None:
        if not self._running:
            return

        try:
            self._running.rtsp.stop()
        except Exception:
            pass

        try:
            self._running.onvif.stop()
        except Exception:
            pass

        try:
            self._running.rtsp.join(timeout=5.0)
        except Exception:
            pass

        try:
            self._running.onvif.join(timeout=5.0)
        except Exception:
            pass

        self._running = None

    def get_latest_frame(self) -> Optional[VideoFrame]:
        if not self._running:
            return None

        latest: Optional[VideoFrame] = None
        try:
            latest = self._running.frame_queue.get_nowait()
        except queue.Empty:
            return None

        # Drain to keep the newest frame (lower latency)
        while True:
            try:
                latest = self._running.frame_queue.get_nowait()
            except queue.Empty:
                break

        return latest

