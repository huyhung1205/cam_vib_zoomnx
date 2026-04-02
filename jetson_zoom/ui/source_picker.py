"""OpenCV-based source picker (name|rtsp_url).

This provides a simple in-app UI for choosing/adding RTSP sources when stdin is not
interactive (common in IDE run configurations) or when users prefer a window-based
prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from jetson_zoom.sources import CameraSource, load_sources, save_sources


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for the source picker UI. "
            "On Windows: pip install opencv-python. "
            "On Jetson: sudo apt-get install python3-opencv."
        ) from e
    return cv2


def _draw_lines(cv2, title: str, lines: Sequence[str], *, w: int = 1100, h: int = 700):
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "numpy is required for the source picker UI. "
            "On Windows: pip install numpy (opencv-python usually pulls it). "
            "On Jetson: sudo apt-get install python3-numpy."
        ) from e

    img = np.zeros((h, w, 3), dtype=np.uint8)
    y = 40
    cv2.putText(img, title, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    y += 40
    for line in lines:
        if y > h - 30:
            break
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        y += 28
    return img


def _text_input(
    cv2,
    window_name: str,
    prompt: str,
    *,
    initial: str = "",
    max_len: int = 300,
) -> Optional[str]:
    value = initial
    hint = "Enter: OK | Esc: Huy | Backspace: Xoa"

    while True:
        img = _draw_lines(
            cv2,
            "JetsonZoom - Nhap thong tin",
            [
                prompt,
                "",
                value if value else "(dang nhap...)",
                "",
                hint,
            ],
        )
        cv2.imshow(window_name, img)

        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue

        if key in (27,):  # ESC
            return None
        if key in (10, 13):  # Enter
            return value.strip()
        if key in (8, 127):  # Backspace
            value = value[:-1]
            continue

        if 32 <= key <= 126 and len(value) < max_len:
            value += chr(key)


def pick_source_opencv(sources_file: Path) -> Optional[CameraSource]:
    """Pick/add a source in an OpenCV window. Returns chosen source or None."""
    cv2 = _import_cv2()
    window_name = "JetsonZoom - Source Picker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            sources = load_sources(sources_file)

            lines = []
            if not sources:
                lines.append("(Chua co source nao trong sources.txt)")
            else:
                for idx, s in enumerate(sources[:9], start=1):
                    lines.append(f"{idx}. {s.name}  ->  {s.rtsp_url}")

            lines += [
                "",
                "Phim:",
                "  1..9 : chon source",
                "  N    : them/ghi de source",
                "  Q/Esc: thoat",
            ]

            img = _draw_lines(cv2, f"Sources: {sources_file.name}", lines)
            cv2.imshow(window_name, img)

            key = cv2.waitKey(50) & 0xFF
            if key == 255:
                continue

            if key in (27, ord("q"), ord("Q")):
                return None

            if ord("1") <= key <= ord("9"):
                idx = int(chr(key)) - 1
                if 0 <= idx < len(sources):
                    return sources[idx]
                continue

            if key in (ord("n"), ord("N")):
                name = _text_input(cv2, window_name, "Nhap NAME (ten camera):")
                if not name:
                    continue
                rtsp = _text_input(cv2, window_name, "Nhap RTSP URL:")
                if not rtsp:
                    continue

                # Upsert by name
                updated = [s for s in sources if s.name.strip().lower() != name.lower()]
                updated.append(CameraSource(name=name, rtsp_url=rtsp))
                updated.sort(key=lambda s: s.name.lower())
                save_sources(sources_file, updated)
                continue
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
