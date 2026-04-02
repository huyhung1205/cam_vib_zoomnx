from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os


@dataclass
class AppState:
    # UI
    ui: str = "qt"  # qt|opencv

    # Source selection
    sources_file: Optional[str] = None
    selected_source_name: Optional[str] = None

    # Connection inputs
    host: str = ""
    onvif_port: int = 80
    username: str = ""
    password: str = ""

    # RTSP
    auto_rtsp: bool = True
    rtsp_url: str = ""


def default_state_path() -> Path:
    # Cross-platform user state location
    base = Path.home() / ".jetsonzoom"
    return base / "state.json"


def state_path_from_env() -> Path:
    value = os.getenv("STATE_FILE", "").strip()
    if value:
        return Path(value)
    return default_state_path()


def load_state(path: Path) -> Optional[AppState]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None

        # Only accept known keys (forward compatible)
        known: Dict[str, Any] = {}
        for field in AppState.__dataclass_fields__.keys():  # type: ignore[attr-defined]
            if field in data:
                known[field] = data[field]

        return AppState(**known)
    except Exception:
        return None


def save_state(path: Path, state: AppState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

