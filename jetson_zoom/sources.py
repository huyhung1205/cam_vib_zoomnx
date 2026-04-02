"""Camera source registry (name -> RTSP URL).

This module provides a simple, git-friendly source list stored in a text file.

File format (UTF-8):
  - Empty lines ignored
  - Lines starting with # are comments
  - Each entry: name|rtsp_url
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CameraSource:
    name: str
    rtsp_url: str


def load_sources(path: Path) -> List[CameraSource]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    sources: List[CameraSource] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "|" not in line:
            continue

        name, rtsp_url = line.split("|", 1)
        name = name.strip()
        rtsp_url = rtsp_url.strip()
        if not name or not rtsp_url:
            continue

        sources.append(CameraSource(name=name, rtsp_url=rtsp_url))

    return sources


def save_sources(path: Path, sources: Iterable[CameraSource]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep a stable, readable format
    lines = [
        "# JetsonZoom sources (name|rtsp_url)",
        "# Lưu ý: nếu RTSP URL có user/pass thì cân nhắc KHÔNG commit lên git.",
        "# Example:",
        "# cam1|rtsp://user:pass@192.168.1.70:554/stream",
        "",
    ]
    for source in sources:
        lines.append(f"{source.name}|{source.rtsp_url}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def find_source(sources: List[CameraSource], name: str) -> Optional[CameraSource]:
    needle = name.strip().lower()
    for source in sources:
        if source.name.strip().lower() == needle:
            return source
    return None


def choose_source_interactive(
    sources_file: Path,
    *,
    allow_new: bool = True,
) -> Tuple[Optional[CameraSource], List[CameraSource]]:
    sources = load_sources(sources_file)

    while True:
        print("")
        print(f"JetsonZoom - chọn RTSP source từ: {sources_file}")
        print("-" * 60)
        if not sources:
            print("(Chưa có source nào trong file)")
        else:
            for idx, s in enumerate(sources, start=1):
                print(f"{idx}. {s.name}  ->  {s.rtsp_url}")
        print("-" * 60)
        print("Nhập số để chọn, 'n' để thêm mới, Enter để bỏ qua:")

        choice = input("> ").strip().lower()
        if choice == "":
            return None, sources

        if allow_new and choice in {"n", "new", "add"}:
            name = input("Name: ").strip()
            rtsp_url = input("RTSP: ").strip()
            if not name or not rtsp_url:
                print("Tên/RTSP không hợp lệ, thử lại.")
                continue

            # Upsert by name
            updated = [s for s in sources if s.name.strip().lower() != name.lower()]
            updated.append(CameraSource(name=name, rtsp_url=rtsp_url))
            updated.sort(key=lambda s: s.name.lower())
            save_sources(sources_file, updated)
            sources = updated
            print("Đã lưu source.")
            continue

        try:
            idx = int(choice)
        except ValueError:
            print("Lựa chọn không hợp lệ.")
            continue

        if 1 <= idx <= len(sources):
            return sources[idx - 1], sources

        print("Số ngoài phạm vi, thử lại.")
