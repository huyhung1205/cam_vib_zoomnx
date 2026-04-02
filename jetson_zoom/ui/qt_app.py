from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

from jetson_zoom.config import ApplicationConfig
from jetson_zoom.logger import get_logger
from jetson_zoom.sources import CameraSource, load_sources, save_sources
from jetson_zoom.state import AppState, load_state, save_state
from jetson_zoom.ui.controller import AppController


def _import_qt():
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets
    except Exception:
        pass

    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyQt không khả dụng. "
            "Windows: pip install PyQt5. "
            "Jetson: sudo apt-get install python3-pyqt5 (và dùng venv --system-site-packages)."
        ) from e


def _qt_enum(QtCore, group_name: str, value_name: str):
    group = getattr(QtCore.Qt, group_name, QtCore.Qt)
    return getattr(group, value_name)


def _bgr_to_qimage(QtGui, image_bgr):
    # image_bgr is a numpy.ndarray (H,W,3) BGR
    h, w = image_bgr.shape[:2]
    bytes_per_line = 3 * w
    image_rgb = image_bgr[:, :, ::-1].copy()
    fmt = getattr(getattr(QtGui.QImage, "Format", QtGui.QImage), "Format_RGB888")
    return QtGui.QImage(
        image_rgb.data,
        w,
        h,
        bytes_per_line,
        fmt,
    )


@dataclass
class UiPaths:
    sources_file: Path
    state_file: Path


class MainWindow:
    def __init__(self, paths: UiPaths, initial_config: ApplicationConfig) -> None:
        QtCore, QtGui, QtWidgets = _import_qt()

        self.QtCore = QtCore
        self.QtGui = QtGui
        self.QtWidgets = QtWidgets

        self.logger = get_logger("JetsonZoom.UI")
        self.paths = paths
        self.controller = AppController()
        self._saving = False

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("JetsonZoom")

        # Debounced state save (must exist before `_build_ui()` triggers callbacks)
        self._save_timer = QtCore.QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save_state_from_ui)

        self._build_ui()
        self._load_sources()
        self._apply_config_to_inputs(initial_config)
        self._load_state_into_ui()

        self._timer = QtCore.QTimer()
        self._timer.setInterval(15)  # ~60 fps UI refresh
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.window.destroyed.connect(lambda: self.controller.stop())

        # Ensure threads stop when the window is closed
        def _on_close(event):
            try:
                self.controller.stop()
            finally:
                event.accept()

        self.window.closeEvent = _on_close

    def show(self) -> None:
        self.window.show()

    # UI ---------------------------------------------------------------------

    def _build_ui(self) -> None:
        QtCore, QtGui, QtWidgets = self.QtCore, self.QtGui, self.QtWidgets

        wheel_event_type = getattr(getattr(QtCore.QEvent, "Type", QtCore.QEvent), "Wheel")
        mouse_press_type = getattr(getattr(QtCore.QEvent, "Type", QtCore.QEvent), "MouseButtonPress")
        mouse_release_type = getattr(getattr(QtCore.QEvent, "Type", QtCore.QEvent), "MouseButtonRelease")
        mouse_move_type = getattr(getattr(QtCore.QEvent, "Type", QtCore.QEvent), "MouseMove")

        class _VideoInputFilter(QtCore.QObject):
            def __init__(self, on_wheel, on_press, on_move, on_release):
                super().__init__()
                self._on_wheel = on_wheel
                self._on_press = on_press
                self._on_move = on_move
                self._on_release = on_release

            def eventFilter(self, obj, event):
                try:
                    if event.type() == wheel_event_type:
                        return bool(self._on_wheel(event))
                    if event.type() == mouse_press_type:
                        return bool(self._on_press(event))
                    if event.type() == mouse_move_type:
                        return bool(self._on_move(event))
                    if event.type() == mouse_release_type:
                        return bool(self._on_release(event))
                except Exception:
                    return False
                return False

        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)

        # Left: controls
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        group_source = QtWidgets.QGroupBox("Nguồn camera (name + RTSP)")
        left.addWidget(group_source)
        form = QtWidgets.QFormLayout(group_source)

        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.currentIndexChanged.connect(self._on_source_selected)
        form.addRow("Source", self.combo_source)

        self.input_name = QtWidgets.QLineEdit()
        form.addRow("Name", self.input_name)

        self.input_rtsp = QtWidgets.QLineEdit()
        self.input_rtsp.setPlaceholderText("rtsp://user:pass@host:554/...")
        self.input_rtsp.editingFinished.connect(self._on_rtsp_edited)
        form.addRow("RTSP", self.input_rtsp)

        self.check_auto_rtsp = QtWidgets.QCheckBox("Tự sinh RTSP từ Host/User/Pass (tắt để sửa)")
        self.check_auto_rtsp.setChecked(True)
        self.check_auto_rtsp.toggled.connect(self._on_auto_rtsp_toggled)
        form.addRow("", self.check_auto_rtsp)

        row_buttons = QtWidgets.QHBoxLayout()
        self.btn_save_source = QtWidgets.QPushButton("Lưu source")
        self.btn_save_source.clicked.connect(self._on_save_source)
        self.btn_new_source = QtWidgets.QPushButton("New")
        self.btn_new_source.clicked.connect(self._on_new_source)
        row_buttons.addWidget(self.btn_save_source)
        row_buttons.addWidget(self.btn_new_source)
        form.addRow("", row_buttons)

        group_onvif = QtWidgets.QGroupBox("ONVIF / PTZ")
        left.addWidget(group_onvif)
        form2 = QtWidgets.QFormLayout(group_onvif)

        self.input_host = QtWidgets.QLineEdit()
        self.input_host.textChanged.connect(lambda: self._on_connection_field_changed())
        form2.addRow("Host", self.input_host)

        self.input_onvif_port = QtWidgets.QSpinBox()
        self.input_onvif_port.setRange(1, 65535)
        self.input_onvif_port.setValue(80)
        self.input_onvif_port.valueChanged.connect(lambda: self._schedule_save())
        form2.addRow("Port", self.input_onvif_port)

        self.input_user = QtWidgets.QLineEdit()
        self.input_user.textChanged.connect(lambda: self._on_connection_field_changed())
        form2.addRow("User", self.input_user)

        self.input_pass = QtWidgets.QLineEdit()
        echo_mode_group = getattr(QtWidgets.QLineEdit, "EchoMode", QtWidgets.QLineEdit)
        self.input_pass.setEchoMode(getattr(echo_mode_group, "Password"))
        self.input_pass.textChanged.connect(lambda: self._on_connection_field_changed())
        form2.addRow("Pass", self.input_pass)

        group_run = QtWidgets.QGroupBox("Chạy / Kết nối")
        left.addWidget(group_run)
        row_run = QtWidgets.QVBoxLayout(group_run)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        row_run.addWidget(self.btn_connect)
        row_run.addWidget(self.btn_disconnect)

        self.label_status = QtWidgets.QLabel("Status: idle")
        self.label_status.setWordWrap(True)
        row_run.addWidget(self.label_status)

        group_zoom = QtWidgets.QGroupBox("Zoom (quang học qua ONVIF)")
        left.addWidget(group_zoom)
        z = QtWidgets.QVBoxLayout(group_zoom)

        self.check_hold = QtWidgets.QCheckBox("Giữ nút để zoom (khuyến nghị)")
        self.check_hold.setToolTip(
            "Bật: nhấn-giữ để zoom liên tục; nhả để STOP.\n"
            "Click nhanh (dưới ~0.2s) sẽ tự zoom theo nhịp (Thời lượng)."
        )
        self.check_hold.setChecked(True)
        self.check_hold.toggled.connect(self._on_hold_toggled)
        z.addWidget(self.check_hold)

        self._hold_direction: Optional[str] = None
        self._hold_started_at: Optional[float] = None
        self._hold_threshold_s: float = 0.20

        self.slider_velocity = QtWidgets.QSlider(_qt_enum(QtCore, "Orientation", "Horizontal"))
        self.slider_velocity.setRange(10, 100)  # 0.10 .. 1.00
        self.slider_velocity.setValue(50)
        self.slider_velocity.setToolTip("Vận tốc tương đối (0.10 = chậm, 1.00 = nhanh).")
        self.label_velocity = QtWidgets.QLabel("Tốc độ (Velocity): 0.50")
        self.slider_velocity.valueChanged.connect(
            lambda v: self.label_velocity.setText(f"Tốc độ (Velocity): {v/100.0:.2f}")
        )
        z.addWidget(self.label_velocity)
        z.addWidget(self.slider_velocity)

        self.spin_duration = QtWidgets.QSpinBox()
        self.spin_duration.setRange(50, 5000)
        self.spin_duration.setValue(500)
        self.spin_duration.setToolTip("Chế độ click/burst: gửi lệnh N mili-giây rồi STOP.")
        z.addWidget(QtWidgets.QLabel("Thời lượng (ms)"))
        z.addWidget(self.spin_duration)

        self.spin_wheel_duration = QtWidgets.QSpinBox()
        self.spin_wheel_duration.setRange(50, 2000)
        self.spin_wheel_duration.setValue(200)
        self.spin_wheel_duration.setToolTip("Mỗi nấc con lăn sẽ zoom 1 nhịp với thời lượng này, rồi STOP.")
        z.addWidget(QtWidgets.QLabel("Con lăn: bước (ms)"))
        z.addWidget(self.spin_wheel_duration)

        row_zoom_btns = QtWidgets.QHBoxLayout()
        self.btn_zoom_in = QtWidgets.QPushButton("Zoom In")
        self.btn_zoom_out = QtWidgets.QPushButton("Zoom Out")
        self.btn_zoom_stop = QtWidgets.QPushButton("Stop")
        self.btn_zoom_in.pressed.connect(lambda: self._on_zoom_press("in"))
        self.btn_zoom_in.released.connect(lambda: self._on_zoom_release())
        self.btn_zoom_in.clicked.connect(lambda: self._on_zoom_click("in"))

        self.btn_zoom_out.pressed.connect(lambda: self._on_zoom_press("out"))
        self.btn_zoom_out.released.connect(lambda: self._on_zoom_release())
        self.btn_zoom_out.clicked.connect(lambda: self._on_zoom_click("out"))

        self.btn_zoom_stop.clicked.connect(lambda: self._on_zoom("stop"))
        row_zoom_btns.addWidget(self.btn_zoom_in)
        row_zoom_btns.addWidget(self.btn_zoom_out)
        row_zoom_btns.addWidget(self.btn_zoom_stop)
        z.addLayout(row_zoom_btns)

        group_pt = QtWidgets.QGroupBox("Pan/Tilt (quay ngang/dọc qua ONVIF)")
        left.addWidget(group_pt)
        pt = QtWidgets.QVBoxLayout(group_pt)

        self.check_drag_pt = QtWidgets.QCheckBox("Kéo chuột trên video để quay (giữ chuột, kéo xa = nhanh)")
        self.check_drag_pt.setChecked(True)
        self.check_drag_pt.setToolTip("Giữ chuột trái trên khung video và kéo để điều khiển pan/tilt.\nThả chuột để STOP.")
        pt.addWidget(self.check_drag_pt)

        self.spin_drag_sensitivity = QtWidgets.QSpinBox()
        self.spin_drag_sensitivity.setRange(1, 200)
        self.spin_drag_sensitivity.setValue(20)
        self.spin_drag_sensitivity.setToolTip("Độ nhạy kéo chuột (lớn = nhạy hơn).")
        pt.addWidget(QtWidgets.QLabel("Độ nhạy kéo (1-200)"))
        pt.addWidget(self.spin_drag_sensitivity)

        self.label_drag_vec = QtWidgets.QLabel("Drag PT: idle")
        self.label_drag_vec.setToolTip("Hiển thị vector pan/tilt đang gửi khi kéo chuột.")
        pt.addWidget(self.label_drag_vec)

        self._pt_hold_action: Optional[str] = None
        self._pt_hold_started_at: Optional[float] = None

        self.slider_pt_velocity = QtWidgets.QSlider(_qt_enum(QtCore, "Orientation", "Horizontal"))
        self.slider_pt_velocity.setRange(10, 100)  # 0.10 .. 1.00
        self.slider_pt_velocity.setValue(50)
        self.slider_pt_velocity.setToolTip("Vận tốc pan/tilt tương đối (0.10 = chậm, 1.00 = nhanh).")
        self.label_pt_velocity = QtWidgets.QLabel("Pan/Tilt Velocity: 0.50")
        self.slider_pt_velocity.valueChanged.connect(
            lambda v: self.label_pt_velocity.setText(f"Pan/Tilt Velocity: {v/100.0:.2f}")
        )
        pt.addWidget(self.label_pt_velocity)
        pt.addWidget(self.slider_pt_velocity)

        self.spin_pt_duration = QtWidgets.QSpinBox()
        self.spin_pt_duration.setRange(50, 5000)
        self.spin_pt_duration.setValue(400)
        self.spin_pt_duration.setToolTip("Chế độ click/burst: gửi lệnh pan/tilt N mili-giây rồi STOP.")
        pt.addWidget(QtWidgets.QLabel("Pan/Tilt Duration (ms)"))
        pt.addWidget(self.spin_pt_duration)

        grid = QtWidgets.QGridLayout()
        self.btn_pt_up = QtWidgets.QPushButton("Up")
        self.btn_pt_down = QtWidgets.QPushButton("Down")
        self.btn_pt_left = QtWidgets.QPushButton("Left")
        self.btn_pt_right = QtWidgets.QPushButton("Right")
        self.btn_pt_stop = QtWidgets.QPushButton("Stop PT")

        # Hold handlers
        self.btn_pt_up.pressed.connect(lambda: self._on_pt_press("up"))
        self.btn_pt_up.released.connect(lambda: self._on_pt_release())
        self.btn_pt_down.pressed.connect(lambda: self._on_pt_press("down"))
        self.btn_pt_down.released.connect(lambda: self._on_pt_release())
        self.btn_pt_left.pressed.connect(lambda: self._on_pt_press("left"))
        self.btn_pt_left.released.connect(lambda: self._on_pt_release())
        self.btn_pt_right.pressed.connect(lambda: self._on_pt_press("right"))
        self.btn_pt_right.released.connect(lambda: self._on_pt_release())

        # Click handlers (burst mode only)
        self.btn_pt_up.clicked.connect(lambda: self._on_pt_click("up"))
        self.btn_pt_down.clicked.connect(lambda: self._on_pt_click("down"))
        self.btn_pt_left.clicked.connect(lambda: self._on_pt_click("left"))
        self.btn_pt_right.clicked.connect(lambda: self._on_pt_click("right"))
        self.btn_pt_stop.clicked.connect(lambda: self._on_pt_stop())

        grid.addWidget(self.btn_pt_up, 0, 1)
        grid.addWidget(self.btn_pt_left, 1, 0)
        grid.addWidget(self.btn_pt_stop, 1, 1)
        grid.addWidget(self.btn_pt_right, 1, 2)
        grid.addWidget(self.btn_pt_down, 2, 1)
        pt.addLayout(grid)

        left.addStretch(1)

        # Right: video
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        self.video = QtWidgets.QLabel("Chưa có video")
        self.video.setMinimumSize(640, 360)
        self.video.setAlignment(_qt_enum(QtCore, "AlignmentFlag", "AlignCenter"))
        self.video.setStyleSheet("background: #111; color: #ddd;")
        self.video.setToolTip(
            "Lăn chuột: zoom quang học (ONVIF), nếu camera hỗ trợ.\n"
            "Giữ chuột trái + kéo: quay ngang/dọc (Pan/Tilt), nếu camera hỗ trợ."
        )
        self.video.setMouseTracking(True)
        self._video_dragging = False
        self._video_drag_start_xy: Optional[tuple[float, float]] = None
        self._video_drag_last_sent_at = 0.0
        self._video_drag_last_vec: Optional[tuple[float, float]] = None
        self._video_input_filter = _VideoInputFilter(
            self._on_zoom_wheel,
            self._on_video_mouse_press,
            self._on_video_mouse_move,
            self._on_video_mouse_release,
        )
        self.video.installEventFilter(self._video_input_filter)
        right.addWidget(self.video, 1)

        self.window.setCentralWidget(central)

        self._on_hold_toggled(self.check_hold.isChecked())
        self._on_auto_rtsp_toggled(self.check_auto_rtsp.isChecked())

    def _on_hold_toggled(self, checked: bool) -> None:
        # Duration only applies to burst (click) mode
        self.spin_duration.setEnabled(not checked)
        if hasattr(self, "spin_pt_duration"):
            self.spin_pt_duration.setEnabled(not checked)

    # Sources ----------------------------------------------------------------

    def _load_sources(self) -> None:
        self.sources = load_sources(self.paths.sources_file)
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        self.combo_source.addItem("(New / Custom)", "")
        for s in self.sources:
            self.combo_source.addItem(s.name, s.rtsp_url)
        self.combo_source.blockSignals(False)

    def _on_new_source(self) -> None:
        self.combo_source.setCurrentIndex(0)
        self.input_name.setText("")
        self.input_rtsp.setText("")

    def _on_source_selected(self) -> None:
        name = self.combo_source.currentText()
        rtsp = self.combo_source.currentData()
        if not rtsp:
            return
        # A saved source might be custom; keep RTSP manual by default.
        self.check_auto_rtsp.setChecked(False)
        self.input_name.setText(name)
        self.input_rtsp.setText(rtsp)
        self._sync_connection_from_rtsp(rtsp)
        self._schedule_save()

    def _on_save_source(self) -> None:
        name = self.input_name.text().strip()
        rtsp = self.input_rtsp.text().strip()
        if not name or not rtsp:
            self._set_status("Thiếu Name hoặc RTSP.")
            return

        updated = [s for s in self.sources if s.name.strip().lower() != name.lower()]
        updated.append(CameraSource(name=name, rtsp_url=rtsp))
        updated.sort(key=lambda s: s.name.lower())
        save_sources(self.paths.sources_file, updated)
        self._set_status("Đã lưu sources.txt")
        self._load_sources()
        self._schedule_save()

        # Select the saved item
        for i in range(self.combo_source.count()):
            if self.combo_source.itemText(i).strip().lower() == name.lower():
                self.combo_source.setCurrentIndex(i)
                break

    # Connect / runtime ------------------------------------------------------

    def _apply_config_to_inputs(self, config: ApplicationConfig) -> None:
        self.input_host.setText(config.camera.host)
        self.input_onvif_port.setValue(config.camera.port_onvif)
        self.input_user.setText(config.camera.username)
        self.input_pass.setText(config.camera.password)
        if config.camera.rtsp_url:
            self.input_rtsp.setText(config.camera.rtsp_url)

    def _load_state_into_ui(self) -> None:
        state = load_state(self.paths.state_file)
        if not state:
            return

        # Apply state (do not override non-empty UI values unless state has value)
        if state.host:
            self.input_host.setText(state.host)
        if state.onvif_port:
            self.input_onvif_port.setValue(int(state.onvif_port))
        if state.username:
            self.input_user.setText(state.username)
        if state.password:
            self.input_pass.setText(state.password)

        self.check_auto_rtsp.setChecked(bool(state.auto_rtsp))
        if state.rtsp_url:
            self.input_rtsp.setText(state.rtsp_url)

        # Select last source name if exists
        if state.selected_source_name:
            for i in range(self.combo_source.count()):
                if self.combo_source.itemText(i).strip().lower() == state.selected_source_name.strip().lower():
                    self.combo_source.setCurrentIndex(i)
                    break

        self._on_auto_rtsp_toggled(self.check_auto_rtsp.isChecked())

    def _schedule_save(self) -> None:
        if self._saving:
            return
        self._save_timer.start()

    def _save_state_from_ui(self) -> None:
        self._saving = True
        try:
            state = AppState(
                ui="qt",
                sources_file=str(self.paths.sources_file),
                selected_source_name=self.combo_source.currentText() if self.combo_source.currentIndex() > 0 else None,
                host=self.input_host.text().strip(),
                onvif_port=int(self.input_onvif_port.value()),
                username=self.input_user.text(),
                password=self.input_pass.text(),
                auto_rtsp=bool(self.check_auto_rtsp.isChecked()),
                rtsp_url=self.input_rtsp.text().strip(),
            )
            save_state(self.paths.state_file, state)
        finally:
            self._saving = False

    def _build_config_from_inputs(self) -> ApplicationConfig:
        config = ApplicationConfig.from_env()

        config.camera.host = self.input_host.text().strip() or config.camera.host
        config.camera.port_onvif = int(self.input_onvif_port.value())
        config.camera.username = self.input_user.text().strip()
        config.camera.password = self.input_pass.text()

        if self.check_auto_rtsp.isChecked():
            config.camera.rtsp_url = self._generate_rtsp_url()
        else:
            rtsp = self.input_rtsp.text().strip()
            if rtsp:
                config.camera.rtsp_url = rtsp

        return config

    def _on_connect(self) -> None:
        try:
            config = self._build_config_from_inputs()
            if not config.camera.rtsp_url:
                self._set_status("Bạn chưa nhập RTSP URL.")
                return
            self.controller.start(config)
            self._set_status("Đang chạy...")
            self._schedule_save()
        except Exception as e:
            self._set_status(f"Lỗi connect: {e}")

    def _on_disconnect(self) -> None:
        self.controller.stop()
        self._set_status("Đã dừng.")
        self._schedule_save()

    def _on_zoom(self, action: str) -> None:
        running = self.controller.running
        if not running:
            self._set_status("Chưa connect.")
            return

        velocity = self.slider_velocity.value() / 100.0
        duration_ms = int(self.spin_duration.value())

        if action == "in":
            running.mover.zoom_in(velocity=velocity, duration_ms=duration_ms)
        elif action == "out":
            running.mover.zoom_out(velocity=velocity, duration_ms=duration_ms)
        else:
            running.mover.stop_movement()

    def _on_zoom_press(self, direction: str) -> None:
        if not self.check_hold.isChecked():
            return
        running = self.controller.running
        if not running:
            return
        self._hold_direction = direction
        self._hold_started_at = time.monotonic()
        velocity = self.slider_velocity.value() / 100.0
        if direction == "in":
            running.mover.zoom_in_hold(velocity=velocity)
        else:
            running.mover.zoom_out_hold(velocity=velocity)

    def _on_zoom_release(self) -> None:
        if not self.check_hold.isChecked():
            return
        running = self.controller.running
        if not running:
            return
        started_at = self._hold_started_at
        direction = self._hold_direction
        self._hold_started_at = None
        self._hold_direction = None

        # If user just "clicked" quickly, treat as a short burst instead of
        # starting and immediately stopping (which often results in no movement).
        if started_at is not None and direction is not None:
            held_s = time.monotonic() - started_at
            if held_s < self._hold_threshold_s:
                velocity = self.slider_velocity.value() / 100.0
                duration_ms = int(self.spin_duration.value())
                if direction == "in":
                    running.mover.zoom_in(velocity=velocity, duration_ms=duration_ms)
                else:
                    running.mover.zoom_out(velocity=velocity, duration_ms=duration_ms)
                return

        running.mover.stop_movement()

    def _on_zoom_click(self, direction: str) -> None:
        # In hold mode, click is already handled by press/release.
        if self.check_hold.isChecked():
            return
        self._on_zoom(direction)

    def _on_zoom_wheel(self, event) -> bool:
        running = self.controller.running
        if not running:
            return False

        try:
            delta_y = int(event.angleDelta().y())
        except Exception:
            return False

        if delta_y == 0:
            return False

        # Standard mouse wheel notch is 120. Trackpads may emit smaller deltas.
        steps = int(delta_y / 120) if abs(delta_y) >= 120 else (1 if delta_y > 0 else -1)
        steps = max(-6, min(6, steps))

        velocity = self.slider_velocity.value() / 100.0
        duration_ms = int(self.spin_wheel_duration.value())

        duration_ms = min(2000, max(50, duration_ms * abs(steps)))
        if steps > 0:
            running.mover.zoom_in(velocity=velocity, duration_ms=duration_ms)
        else:
            running.mover.zoom_out(velocity=velocity, duration_ms=duration_ms)

        try:
            event.accept()
        except Exception:
            pass
        return True

    def _event_xy(self, event) -> tuple[float, float]:
        try:
            p = event.position()
            return float(p.x()), float(p.y())
        except Exception:
            p = event.pos()
            return float(p.x()), float(p.y())

    def _on_video_mouse_press(self, event) -> bool:
        if not hasattr(self, "check_drag_pt") or not self.check_drag_pt.isChecked():
            return False

        running = self.controller.running
        if not running:
            return False

        left_button = _qt_enum(self.QtCore, "MouseButton", "LeftButton")
        try:
            if event.button() != left_button:
                return False
        except Exception:
            return False

        self._video_dragging = True
        self._video_drag_start_xy = self._event_xy(event)
        self._video_drag_last_sent_at = 0.0
        self._video_drag_last_vec = None
        if hasattr(self, "label_drag_vec"):
            self.label_drag_vec.setText("Drag PT: start")

        try:
            event.accept()
        except Exception:
            pass

        # Ensure we don't keep moving from a previous pan/tilt command.
        try:
            running.mover.stop_pan_tilt()
        except Exception:
            pass
        return True

    def _on_video_mouse_move(self, event) -> bool:
        if not getattr(self, "_video_dragging", False):
            return False

        running = self.controller.running
        if not running:
            return False

        left_button = _qt_enum(self.QtCore, "MouseButton", "LeftButton")
        try:
            if not (event.buttons() & left_button):
                return False
        except Exception:
            return False

        start = self._video_drag_start_xy
        if not start:
            return False

        x, y = self._event_xy(event)
        dx = x - start[0]
        dy = y - start[1]

        sensitivity_raw = self.spin_drag_sensitivity.value() if hasattr(self, "spin_drag_sensitivity") else 20
        sensitivity = float(sensitivity_raw) / 2000.0
        sensitivity = max(0.0005, min(0.2, sensitivity))

        pan_x = max(-1.0, min(1.0, dx * sensitivity))
        pan_y = max(-1.0, min(1.0, -dy * sensitivity))

        # Deadzone near center
        if abs(pan_x) < 0.02:
            pan_x = 0.0
        if abs(pan_y) < 0.02:
            pan_y = 0.0

        now = time.monotonic()
        last_vec = self._video_drag_last_vec
        if last_vec is not None:
            if now - float(self._video_drag_last_sent_at) < 0.03:
                if abs(pan_x - last_vec[0]) < 0.02 and abs(pan_y - last_vec[1]) < 0.02:
                    return True

        self._video_drag_last_sent_at = now
        self._video_drag_last_vec = (pan_x, pan_y)
        if hasattr(self, "label_drag_vec"):
            self.label_drag_vec.setText(f"Drag PT: ({pan_x:+.2f}, {pan_y:+.2f})")

        try:
            velocity = self.slider_pt_velocity.value() / 100.0
        except Exception:
            velocity = 0.5

        try:
            if pan_x == 0.0 and pan_y == 0.0:
                running.mover.stop_pan_tilt()
            else:
                running.onvif.queue_pan_tilt_command(
                    pan_x=pan_x,
                    pan_y=pan_y,
                    velocity=velocity,
                    hold=True,
                )
        except Exception:
            return False

        try:
            event.accept()
        except Exception:
            pass
        return True

    def _on_video_mouse_release(self, event) -> bool:
        if not getattr(self, "_video_dragging", False):
            return False

        running = self.controller.running
        if not running:
            return False

        left_button = _qt_enum(self.QtCore, "MouseButton", "LeftButton")
        try:
            if event.button() != left_button:
                return False
        except Exception:
            return False

        self._video_dragging = False
        self._video_drag_start_xy = None
        self._video_drag_last_vec = None
        if hasattr(self, "label_drag_vec"):
            self.label_drag_vec.setText("Drag PT: idle")

        try:
            running.mover.stop_pan_tilt()
        except Exception:
            pass

        try:
            event.accept()
        except Exception:
            pass
        return True

    # Pan/Tilt ---------------------------------------------------------------

    def _on_pt_press(self, action: str) -> None:
        if not self.check_hold.isChecked():
            return
        running = self.controller.running
        if not running:
            return
        self._pt_hold_action = action
        self._pt_hold_started_at = time.monotonic()
        velocity = self.slider_pt_velocity.value() / 100.0
        if action == "left":
            running.mover.pan_left_hold(velocity=velocity)
        elif action == "right":
            running.mover.pan_right_hold(velocity=velocity)
        elif action == "up":
            running.mover.tilt_up_hold(velocity=velocity)
        else:
            running.mover.tilt_down_hold(velocity=velocity)

    def _on_pt_release(self) -> None:
        if not self.check_hold.isChecked():
            return
        running = self.controller.running
        if not running:
            return
        started_at = self._pt_hold_started_at
        action = self._pt_hold_action
        self._pt_hold_started_at = None
        self._pt_hold_action = None

        # Quick-click: convert hold into a short burst
        if started_at is not None and action is not None:
            held_s = time.monotonic() - started_at
            if held_s < self._hold_threshold_s:
                velocity = self.slider_pt_velocity.value() / 100.0
                duration_ms = int(self.spin_pt_duration.value())
                if action == "left":
                    running.mover.pan_left(velocity=velocity, duration_ms=duration_ms)
                elif action == "right":
                    running.mover.pan_right(velocity=velocity, duration_ms=duration_ms)
                elif action == "up":
                    running.mover.tilt_up(velocity=velocity, duration_ms=duration_ms)
                else:
                    running.mover.tilt_down(velocity=velocity, duration_ms=duration_ms)
                return

        running.mover.stop_pan_tilt()

    def _on_pt_click(self, action: str) -> None:
        # In hold mode, click is handled by press/release.
        if self.check_hold.isChecked():
            return
        running = self.controller.running
        if not running:
            return
        velocity = self.slider_pt_velocity.value() / 100.0
        duration_ms = int(self.spin_pt_duration.value())
        if action == "left":
            running.mover.pan_left(velocity=velocity, duration_ms=duration_ms)
        elif action == "right":
            running.mover.pan_right(velocity=velocity, duration_ms=duration_ms)
        elif action == "up":
            running.mover.tilt_up(velocity=velocity, duration_ms=duration_ms)
        else:
            running.mover.tilt_down(velocity=velocity, duration_ms=duration_ms)

    def _on_pt_stop(self) -> None:
        running = self.controller.running
        if not running:
            return
        running.mover.stop_pan_tilt()

    def _on_connection_field_changed(self) -> None:
        if self.check_auto_rtsp.isChecked():
            self.input_rtsp.setText(self._generate_rtsp_url())
        self._schedule_save()

    def _on_auto_rtsp_toggled(self, checked: bool) -> None:
        # When auto mode is on, RTSP is derived from connection fields
        self.input_rtsp.setReadOnly(checked)
        if checked:
            self.input_rtsp.setText(self._generate_rtsp_url())
        self._schedule_save()

    def _on_rtsp_edited(self) -> None:
        # If user edited RTSP while in auto mode, switch to manual mode.
        if self.check_auto_rtsp.isChecked():
            self.check_auto_rtsp.setChecked(False)
            return

        rtsp = self.input_rtsp.text().strip()
        if rtsp:
            self._sync_connection_from_rtsp(rtsp)
        self._schedule_save()

    def _generate_rtsp_url(self) -> str:
        from urllib.parse import quote

        host = self.input_host.text().strip()
        user = self.input_user.text().strip()
        password = self.input_pass.text()

        if not host:
            return ""

        auth = ""
        if user:
            auth = f"{quote(user, safe='')}:{quote(password, safe='')}@"

        # Default IMOU/Dahua style path (user can switch off auto and edit if needed)
        return f"rtsp://{auth}{host}:554/cam/realmonitor?channel=1&subtype=0"

    def _sync_connection_from_rtsp(self, rtsp_url: str) -> None:
        from urllib.parse import urlparse, unquote

        parsed = urlparse(rtsp_url)
        if parsed.hostname:
            self.input_host.setText(parsed.hostname)
        if parsed.username:
            self.input_user.setText(unquote(parsed.username))
        if parsed.password is not None:
            self.input_pass.setText(unquote(parsed.password))

    # Tick -------------------------------------------------------------------

    def _on_tick(self) -> None:
        frame = self.controller.get_latest_frame()
        if frame is not None:
            qimg = _bgr_to_qimage(self.QtGui, frame.image)
            pix = self.QtGui.QPixmap.fromImage(qimg)
            self.video.setPixmap(
                pix.scaled(
                    self.video.size(),
                    _qt_enum(self.QtCore, "AspectRatioMode", "KeepAspectRatio"),
                )
            )

        running = self.controller.running
        if running:
            onvif_ready = running.onvif.is_ready()
            zoom_x = running.mover.get_zoom_level()
            zoom_cap = running.onvif.is_zoom_supported()
            pt_cap = running.onvif.is_pan_tilt_supported()
            last_error = running.onvif.get_last_error()
            err_text = f" | Err: {last_error}" if last_error else ""
            zoom_text = f"{zoom_x:.2f}" if zoom_x is not None else "n/a"
            zoom_cap_text = "OK" if zoom_cap is True else ("NO" if zoom_cap is False else "?")
            pt_cap_text = "OK" if pt_cap is True else ("NO" if pt_cap is False else "?")
            self.label_status.setText(
                f"Status: running | ONVIF: {'OK' if onvif_ready else 'NO'} | ZoomCap: {zoom_cap_text} | PanTiltCap: {pt_cap_text} | Zoom.x: {zoom_text}{err_text}"
            )

    def _set_status(self, text: str) -> None:
        self.label_status.setText(f"Status: {text}")


def run_qt_ui(sources_file: Path, initial_config: ApplicationConfig) -> int:
    QtCore, QtGui, QtWidgets = _import_qt()

    app = QtWidgets.QApplication([])
    from jetson_zoom.state import state_path_from_env

    win = MainWindow(
        UiPaths(sources_file=sources_file, state_file=state_path_from_env()),
        initial_config,
    )
    win.show()
    return app.exec()
