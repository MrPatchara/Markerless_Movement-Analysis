"""
หน้าต่างหลัก – SLS Analysis, Force, 2D/3D View
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QStatusBar, QMessageBox,
    QTextEdit, QFrame, QRadioButton, QLineEdit, QComboBox, QSpinBox,
    QDialog, QDialogButtonBox, QStyle,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from biomech.c3d_loader import load_motion_file, C3DData
from biomech.sls_analysis import (
    analyze_sls_limb_stability,
    analyze_pelvis_stability,
    analyze_trunk_stability,
    analyze_movement_strategy,
    analyze_shock_absorption,
    format_limb_stability_report,
    format_pelvis_stability_report,
    format_trunk_stability_report,
    format_movement_strategy_report,
    format_shock_absorption_report,
    find_frame_max_knee_flexion,
    get_knee_flexion_deg,
    get_ls_frontal_2d_geometry,
    get_ms_sagittal_2d_geometry,
    get_sa_sagittal_2d_geometry,
    get_ps_frontal_2d_geometry,
    get_ts_frontal_2d_geometry,
)
from biomech.force_filter import filter_force_plate_data, filter_cop_data
from biomech.grf_metrics import grf_inclination_deg

from biomech_app.theme import COLORS
from biomech_app.workers import LoadWorker
from biomech_app.widgets import (
    FPKPADiagramWidget,
    ResizeDebounceWidget,
    SkeletonViewerWidget,
)


class BiomechMainWindow(QMainWindow):
    """หน้าต่างหลัก – โหลดไฟล์, Run SLS, แสดงผล"""

    def __init__(self):
        super().__init__()
        self.data: Optional[C3DData] = None
        self.current_file_path: Optional[str] = None
        self.current_frame = 0
        self._load_worker: Optional[LoadWorker] = None
        self.video_front_path: Optional[str] = None
        self.video_side_path: Optional[str] = None
        self._last_analysis_frame: Optional[int] = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Markerless Movement Analysis")
        # ตั้งไอคอนโปรแกรมบน title bar จากไฟล์ logo2.png
        logo_path = Path(__file__).resolve().parent.parent / "logo2.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        self.setMinimumSize(900, 620)
        self.resize(1100, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Toolbar
        toolbar = QHBoxLayout()
        open_btn = QPushButton("📂  Open Motion File")
        open_btn.setObjectName("openButton")
        open_btn.setMinimumHeight(36)
        open_btn.clicked.connect(self.open_file)
        open_btn.setToolTip("เปิดไฟล์ C3D (กด Close ก่อนเพื่อเปิดไฟล์ใหม่)")
        toolbar.addWidget(open_btn)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("closeButton")
        close_btn.setMinimumHeight(36)
        close_btn.clicked.connect(self.close_file)
        close_btn.setEnabled(False)
        close_btn.setToolTip("ปิดไฟล์ C3D ปัจจุบัน เพื่อเปิดไฟล์ใหม่ได้")
        toolbar.addWidget(close_btn)
        self.open_btn_ref = open_btn
        self.close_btn_ref = close_btn
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 13px;")
        toolbar.addWidget(self.file_label)
        toolbar.addStretch()
        self.frame_play_btn = QPushButton()
        self.frame_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.frame_play_btn.setText("")
        self.frame_play_btn.setEnabled(False)
        self.frame_play_btn.setCheckable(True)
        self.frame_play_btn.setToolTip("เล่น/หยุด (Play/Pause)")
        self.frame_play_btn.clicked.connect(self._on_frame_play_toggled)
        toolbar.addWidget(self.frame_play_btn)
        toolbar.addWidget(QLabel("ความเร็ว:"))
        self.play_speed_combo = QComboBox()
        self.play_speed_combo.addItems(["x1", "x2.5", "x5"])
        self.play_speed_combo.setCurrentIndex(0)
        self.play_speed_combo.setToolTip("ความเร็วเล่น (เทียบ frame rate ต้นฉบับ)")
        self.play_speed_combo.setMinimumWidth(56)
        self.play_speed_combo.currentIndexChanged.connect(self._on_play_speed_changed)
        toolbar.addWidget(self.play_speed_combo)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_timer)
        front_vid_btn = QPushButton("Video Front view")
        front_vid_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        front_vid_btn.clicked.connect(lambda: self._pick_video("front"))
        toolbar.addWidget(front_vid_btn)
        side_vid_btn = QPushButton("Video Side view")
        side_vid_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        side_vid_btn.clicked.connect(lambda: self._pick_video("side"))
        toolbar.addWidget(side_vid_btn)
        about_btn = QPushButton("About")
        about_btn.setObjectName("aboutButton")
        about_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        about_btn.clicked.connect(self._show_about)
        about_btn.setToolTip("เกี่ยวกับโปรแกรม")
        toolbar.addWidget(about_btn)
        layout.addLayout(toolbar)

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        info_group = QGroupBox("File Info")
        info_group.setMaximumHeight(105)
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(10, 8, 10, 8)
        self.info_label = QLabel("Load a C3D file to see details.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        # แถวเลือกแบบทดสอบ (Squat / Hop) อยู่ถัดจาก File Info
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("แบบทดสอบ:"))
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems(["Single Leg Squat", "Single Leg Hop"])
        self.test_type_combo.setToolTip(
            "Squat: ยืนบนแผ่นแล้วย่อ | Hop: กระโดดเข้าแผ่น – เฟรมเข่างอสุดจะนับเฉพาะตอนเท้าแตะแผ่น"
        )
        self.test_type_combo.currentIndexChanged.connect(self._on_test_type_changed)
        test_row.addWidget(self.test_type_combo)
        test_row.addStretch()
        left_layout.addLayout(test_row)

        analysis_group = QGroupBox("Analysis (2D)")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("ขา:"))
        self.side_left_radio = QRadioButton("L")
        self.side_right_radio = QRadioButton("R")
        self.side_left_radio.setChecked(True)
        control_row.addWidget(self.side_left_radio)
        control_row.addWidget(self.side_right_radio)
        self.analysis_type_combo = QComboBox()
        self._populate_analysis_types()
        control_row.addWidget(self.analysis_type_combo)
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.run_sls_analysis)
        run_btn.setEnabled(False)
        self.run_sls_btn_ref = run_btn
        control_row.addWidget(run_btn)
        control_row.addWidget(QLabel("เฟรม:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.frame_spin.setValue(0)
        self.frame_spin.setToolTip("เปลี่ยนเฟรมแล้วกด Prev/Next เพื่ออัปเดตผล")
        self.frame_spin.setEnabled(False)
        self.frame_spin.valueChanged.connect(self._on_frame_changed)
        control_row.addWidget(self.frame_spin)
        self.frame_prev_btn = QPushButton("◀")
        self.frame_prev_btn.setEnabled(False)
        self.frame_prev_btn.clicked.connect(self._on_frame_prev)
        control_row.addWidget(self.frame_prev_btn)
        self.frame_next_btn = QPushButton("▶")
        self.frame_next_btn.setEnabled(False)
        self.frame_next_btn.clicked.connect(self._on_frame_next)
        control_row.addWidget(self.frame_next_btn)
        control_row.addStretch()
        analysis_layout.addLayout(control_row)
        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setMinimumHeight(180)
        self.analysis_output.setStyleSheet(
            f"font-family: Consolas, monospace; font-size: 11px; color: {COLORS['text']}; "
            f"background: {COLORS['bg_card']}; padding: 8px;"
        )
        self.analysis_output.setPlaceholderText("Load file and click Run...")
        analysis_layout.addWidget(self.analysis_output, 1)
        left_layout.addWidget(analysis_group, 1)

        force_group = QGroupBox("Data (Force)")
        force_group.setMaximumHeight(95)
        force_layout = QHBoxLayout(force_group)
        force_layout.setContentsMargins(8, 6, 8, 6)
        self.force_output = QTextEdit()
        self.force_output.setReadOnly(True)
        self.force_output.setMaximumHeight(68)
        self.force_output.setStyleSheet(
            f"font-family: Consolas, monospace; font-size: 10px; color: {COLORS['text']}; "
            f"background: {COLORS['bg_card']}; padding: 4px 6px; border: 1px solid {COLORS['bg_elevated']};"
        )
        self.force_output.setPlaceholderText("Run analysis เพื่อแสดงพารามิเตอร์เฟรม")
        force_layout.addWidget(self.force_output, 1)
        left_layout.addWidget(force_group, 0)
        left_panel.setMinimumWidth(420)
        main_row.addWidget(left_panel)

        # Right panel: video + 2D + 3D
        right_panel = QFrame()
        right_panel.setStyleSheet("QFrame { border: 1px solid #415A77; border-radius: 8px; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)

        video_row = QHBoxLayout()
        video_row.setSpacing(8)
        self.video_front_label = QLabel()
        self.video_front_label.setMinimumSize(200, 200)
        self.video_front_label.setFixedHeight(260)
        self.video_front_label.setAlignment(Qt.AlignCenter)
        self.video_front_label.setStyleSheet(
            f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['bg_elevated']}; "
            f"border-radius: 6px; color: {COLORS['text_dim']}; font-size: 11px;"
        )
        self.video_front_label.setText("Front view\n(เลือก .avi)")
        self.video_side_label = QLabel()
        self.video_side_label.setMinimumSize(200, 200)
        self.video_side_label.setFixedHeight(260)
        self.video_side_label.setAlignment(Qt.AlignCenter)
        self.video_side_label.setStyleSheet(
            f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['bg_elevated']}; "
            f"border-radius: 6px; color: {COLORS['text_dim']}; font-size: 11px;"
        )
        self.video_side_label.setText("Side view\n(เลือก .avi)")
        video_row.addWidget(self.video_front_label, 1)
        video_row.addWidget(self.video_side_label, 1)
        right_layout.addLayout(video_row)

        view_row = QHBoxLayout()
        view_row.setSpacing(12)
        fpkpa_group = QGroupBox("2D")
        fpkpa_group.setMinimumWidth(240)
        fpkpa_group.setMinimumHeight(280)
        fpkpa_layout = QVBoxLayout(fpkpa_group)
        fpkpa_layout.setContentsMargins(6, 12, 6, 6)
        self.fpkpa_diagram = FPKPADiagramWidget(fpkpa_group)
        fpkpa_layout.addWidget(self.fpkpa_diagram)
        view_row.addWidget(fpkpa_group, 0)
        viewer_3d_group = QGroupBox("3D")
        viewer_3d_group.setMinimumWidth(280)
        viewer_3d_group.setMinimumHeight(280)
        viewer_3d_layout = QVBoxLayout(viewer_3d_group)
        viewer_3d_layout.setContentsMargins(6, 12, 6, 6)
        self.skeleton_viewer = SkeletonViewerWidget(right_panel)
        viewer_wrapper = ResizeDebounceWidget(self.skeleton_viewer, right_panel)
        viewer_3d_layout.addWidget(viewer_wrapper)
        view_row.addWidget(viewer_3d_group, 1)
        right_layout.addLayout(view_row, 1)
        right_panel.setMinimumWidth(640)
        main_row.addWidget(right_panel)
        layout.addLayout(main_row)

        self.statusBar().showMessage("Ready - Open C3D file to begin")

    def close_file(self):
        """ปิด/รีเซ็ตไฟล์ C3D ปัจจุบัน – เลือกไฟล์ใหม่จะได้ไม่ค้าง"""
        self.data = None
        self.current_file_path = None
        self.current_frame = 0
        self._last_analysis_frame = None
        self.video_front_path = None
        self.video_side_path = None
        self.file_label.setText("No file loaded")
        self.info_label.setText("Load a C3D file to see details.")
        self.skeleton_viewer.reset()
        self.fpkpa_diagram.update_diagram(None)
        self.analysis_output.clear()
        self.force_output.clear()
        self.video_front_label.clear()
        self.video_front_label.setText("Front view\n(เลือก .avi)")
        self.video_side_label.clear()
        self.video_side_label.setText("Side view")
        self.run_sls_btn_ref.setEnabled(False)
        self.frame_spin.setMaximum(0)
        self.frame_spin.setValue(0)
        self.frame_spin.setEnabled(False)
        self.frame_prev_btn.setEnabled(False)
        self.frame_next_btn.setEnabled(False)
        self.frame_play_btn.setEnabled(False)
        self.frame_play_btn.setChecked(False)
        self._play_timer.stop()
        self.open_btn_ref.setEnabled(True)
        self.close_btn_ref.setEnabled(False)
        self.statusBar().showMessage("Ready - Open C3D file to begin")

    def _populate_analysis_types(self):
        """ปรับตัวเลือกการวิเคราะห์ตามแบบทดสอบ (Squat / Hop)."""
        current = self.analysis_type_combo.currentText() if hasattr(self, "analysis_type_combo") else ""
        self.analysis_type_combo.clear()
        base_items = [
            "Limb Stability (LS)",
            "Pelvis Stability (PS)",
            "Trunk Stability (TS)",
        ]
        is_hop = self.test_type_combo.currentText() == "Single Leg Hop"
        items = list(base_items)
        if is_hop:
            items.append("Shock Absorption (SA)")
        items.append("Movement Strategy (MS)")
        self.analysis_type_combo.addItems(items)
        # ตั้ง tooltip ตามแบบทดสอบ
        if is_hop:
            self.analysis_type_combo.setToolTip(
                "LS: FPKPA | PS: PA (pelvis) | TS: TA (trunk) | SA: Shock Absorption (Hop) | MS: α/β (sagittal)"
            )
        else:
            self.analysis_type_combo.setToolTip(
                "LS: FPKPA | PS: PA (pelvis) | TS: TA (trunk) | MS: α/β (sagittal)"
            )
        # พยายามรักษา selection เดิม ถ้าไม่เจอให้ fallback เป็น LS
        idx = self.analysis_type_combo.findText(current)
        if idx < 0:
            idx = 0
        self.analysis_type_combo.setCurrentIndex(idx)

    def _on_test_type_changed(self, index: int):
        """เมื่อเปลี่ยนแบบทดสอบ (Squat/Hop) ให้ปรับตัวเลือก Analysis ให้เหมาะสม."""
        prev = self.analysis_type_combo.currentText() if hasattr(self, "analysis_type_combo") else ""
        self._populate_analysis_types()
        # ถ้าเพิ่งสลับจาก Hop → Squat และก่อนหน้าคือ Shock Absorption ให้เปลี่ยนเป็น Movement Strategy
        if "Shock Absorption" in prev and self.test_type_combo.currentText() == "Single Leg Squat":
            ms_idx = self.analysis_type_combo.findText("Movement Strategy (MS)")
            if ms_idx >= 0:
                self.analysis_type_combo.setCurrentIndex(ms_idx)

    def open_file(self):
        base = Path(__file__).resolve().parent.parent
        default_path = base / "c3d"
        if not default_path.exists():
            default_path = base
        if not default_path.exists():
            default_path = Path.home()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open C3D File",
            str(default_path),
            "C3D / JSON (*.c3d *.json);;C3D (*.c3d);;All Files (*)",
        )
        if path:
            self.load_motion_file(path)

    def _on_load_finished(self, data: C3DData, path: str):
        self._load_worker = None
        self.data = data
        self.current_file_path = path
        self.current_frame = 0
        self.file_label.setText(Path(path).name)
        self.skeleton_viewer.reset()
        self.fpkpa_diagram.update_diagram(None)
        self.analysis_output.clear()
        self.force_output.clear()
        self.video_front_label.clear()
        self.video_front_label.setText("Front view\n(เลือก .avi)")
        self.video_side_label.clear()
        self.video_side_label.setText("Side view")
        self.info_label.setText(
            f"<b>Frames:</b> {data.n_frames}  |  "
            f"<b>Frame rate:</b> {data.frame_rate:.1f} Hz<br>"
            f"<b>Points:</b> {data.n_points}  |  "
            f"<b>Duration:</b> {data.n_frames / data.frame_rate:.2f} s<br>"
            f"<b>Labels:</b> {', '.join(data.labels[:10])}{'...' if len(data.labels) > 10 else ''}"
        )
        self.run_sls_btn_ref.setEnabled(True)
        self.frame_spin.setMaximum(max(0, data.n_frames - 1))
        self.frame_spin.setValue(0)
        self.frame_spin.setEnabled(False)
        self.frame_prev_btn.setEnabled(False)
        self.frame_next_btn.setEnabled(False)
        self.frame_play_btn.setEnabled(False)
        self.frame_play_btn.setChecked(False)
        self._play_timer.stop()
        self.open_btn_ref.setEnabled(False)
        self.close_btn_ref.setEnabled(True)
        self.statusBar().showMessage(f"Loaded: {path}")

    def _on_load_error(self, err: str):
        self._load_worker = None
        self.file_label.setText("No file loaded")
        self.open_btn_ref.setEnabled(True)
        self.close_btn_ref.setEnabled(False)
        QMessageBox.critical(self, "Load Error", f"Could not load file:\n{err}")
        self.statusBar().showMessage(f"Error: {err}")

    def _pick_video(self, which: str):
        base = Path(__file__).resolve().parent.parent
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {which.capitalize()} view video",
            str(base),
            "Video (*.avi *.mp4 *.mov);;All Files (*)",
        )
        if path:
            if which == "front":
                self.video_front_path = path
                self.video_front_label.setText(Path(path).name)
            else:
                self.video_side_path = path
                self.video_side_label.setText(Path(path).name)
            self.video_front_label.setStyleSheet(
                f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['bg_elevated']}; border-radius: 6px;"
            )
            self.video_side_label.setStyleSheet(
                f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['bg_elevated']}; border-radius: 6px;"
            )

    def _show_about(self):
        """แสดงหน้าต่าง About – ข้อมูลโปรเจกต์และการใช้งาน"""
        dlg = QDialog(self)
        dlg.setWindowTitle("About")
        dlg.setMinimumWidth(440)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(16)

        version_label = QLabel("Version: 1.0")
        version_label.setAlignment(Qt.AlignLeft)
        version_label.setStyleSheet("font-size: 11px; color: #778DA9;")
        layout.addWidget(version_label)

        logo_path = Path(__file__).resolve().parent.parent / "logo.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                logo_label = QLabel()
                logo_label.setAlignment(Qt.AlignCenter)
                max_wh = 200
                scaled = pix.scaled(max_wh, max_wh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled)
                layout.addWidget(logo_label)
                layout.addSpacing(8)

        section_style = "font-size: 11px; font-weight: 600; color: #00D4AA; margin-top: 2px;"
        body_style = f"color: {COLORS['text']}; font-size: 12px; line-height: 1.5;"

        # หน่วยงานและระบบ
        p1 = QLabel(
            "โปรแกรมนี้ใช้ร่วมกับอุปกรณ์ \"แผ่นวัดแรง Podium BTS Bioengineering\"<br>"
            "รองรับการทำงานกับระบบ <b>Markerless Motion Capture</b> โดยรับข้อมูลจากไฟล์ C3D"
        )
        p1.setWordWrap(True)
        p1.setStyleSheet(body_style)
        p1.setTextFormat(Qt.RichText)
        layout.addWidget(p1)

        # วัตถุประสงค์
        h2 = QLabel("การใช้งาน")
        h2.setStyleSheet(section_style)
        layout.addWidget(h2)
        p2 = QLabel(
            "วิเคราะห์ท่า <b>Single Leg Squat (SLS)</b> และ <b>Single Leg Hop</b> "
            "จากข้อมูล C3D ที่ได้จากระบบ markerless motion capture ดังนี้<br><br>"
            "<b>1.</b> ประเมินการเคลื่อนไหวตามเกณฑ์อ้างอิง<br>"
            "   • <b>Limb Stability (LS)</b> — มุม FPKPA ในระนาบ frontal<br>"
            "   • <b>Pelvis Stability (PS)</b> — มุม PA (pelvis tilt)<br>"
            "   • <b>Trunk Stability (TS)</b> — มุม TA (trunk tilt)<br>"
            "   • <b>Shock Absorption (SA)</b> — มุม α (knee flexion) ในจังหวะลงพื้น<br>"
            "   • <b>Movement Strategy (MS)</b> — มุม α (knee flexion) และ β (hip–trunk)<br><br>"
            "<b>2.</b> แสดงผลในมุมมอง <b>2D</b> (frontal/sagittal) และ <b>3D</b> พร้อมแกนอ้างอิงและกริด<br><br>"
            "<b>3.</b> ให้คะแนนตามเกณฑ์ และแสดงคำแนะนำเชิงคลินิก/ชีวกลศาสตร์จากการวิเคราะห์"
        )
        p2.setWordWrap(True)
        p2.setStyleSheet(body_style)
        p2.setTextFormat(Qt.RichText)
        layout.addWidget(p2)

        # Developer
        h3 = QLabel("Developer")
        h3.setStyleSheet(section_style)
        layout.addWidget(h3)
        dev = QLabel(
            "Mr.Patchara<br>"
            "Email:  <a href=\"mailto:Patcharaalumaree@gmail.com\">Patcharaalumaree@gmail.com</a><br>"
            "GitHub: <a href=\"https://github.com/MrPatchara\">github.com/MrPatchara</a>"
        )
        dev.setWordWrap(True)
        dev.setStyleSheet(body_style)
        dev.setTextFormat(Qt.RichText)
        dev.setOpenExternalLinks(True)
        layout.addWidget(dev)

        dlg.exec()

    def _show_video_frames(self, c3d_frame_idx: int):
        if not CV2_AVAILABLE or self.data is None:
            return
        c3d_fps = self.data.frame_rate
        time_sec = c3d_frame_idx / c3d_fps
        for path_attr, label_attr in [
            ("video_front_path", "video_front_label"),
            ("video_side_path", "video_side_label"),
        ]:
            path = getattr(self, path_attr, None)
            if not path:
                continue
            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_frame = int(round(time_sec * fps))
                vid_frame = max(0, min(vid_frame, total - 1)) if total > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                lbl = getattr(self, label_attr)
                wd, ht = max(220, lbl.width()), max(240, lbl.height())
                pix = pix.scaled(wd, ht, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)
            except Exception:
                pass

    def load_motion_file(self, path: str):
        if self._load_worker and self._load_worker.isRunning():
            return
        self.statusBar().showMessage(f"Loading {Path(path).name}...")
        self.file_label.setText("Loading...")
        self._load_worker = LoadWorker(path, self)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _update_force_display(self, frame_idx: int):
        ff = getattr(self.data, "force_forces", None)
        fc = getattr(self.data, "force_cops", None)
        side = "L" if self.side_left_radio.isChecked() else "R"
        knee_flex = get_knee_flexion_deg(self.data, frame_idx, side) if self.data is not None else None
        if ff is None or fc is None or frame_idx >= ff.shape[0]:
            if knee_flex is not None:
                self.force_output.setPlainText(f"Frame: {frame_idx}\nKnee flex: {knee_flex:.4f}°\n\nไม่มีข้อมูลแรงจาก C3D")
            else:
                self.force_output.setPlainText("ไม่มีข้อมูลแรงจาก C3D")
            return
        fps = self.data.frame_rate
        mags = np.linalg.norm(ff[frame_idx], axis=1)
        best = int(np.argmax(mags))
        if mags[best] < 10:
            if knee_flex is not None:
                self.force_output.setPlainText(f"Frame: {frame_idx}\nKnee flex: {knee_flex:.4f}°\n\nแรงต่ำกว่า 10 N ในเฟรมนี้")
            else:
                self.force_output.setPlainText("แรงต่ำกว่า 10 N ในเฟรมนี้")
            return
        raw_f = ff[frame_idx, best]
        raw_mag = np.linalg.norm(raw_f)
        raw_cop = fc[frame_idx, best]
        try:
            lp_ff = filter_force_plate_data(ff, fps, cutoff_hz=10.0, order=4)
            lp_fc = filter_cop_data(fc, fps, cutoff_hz=10.0, order=4)
            lp_f = lp_ff[frame_idx, best]
            lp_mag = np.linalg.norm(lp_f)
            lp_cop = lp_fc[frame_idx, best]
        except Exception:
            incl_raw = grf_inclination_deg(raw_f[0], raw_f[1], raw_f[2])
            lines = [
                f"Frame: {frame_idx}",
                *([f"Knee flex: {knee_flex:.4f}°"] if knee_flex is not None else []),
                f"Plate: {best}  Rate: {fps:.0f} Hz",
                "",
                "Raw  |F|: {:.1f} N  GRF: {:.1f}°".format(raw_mag, incl_raw),
                "Filter: scipy ไม่พร้อม",
            ]
            self.force_output.setPlainText("\n".join(lines))
            return
        incl_raw = grf_inclination_deg(raw_f[0], raw_f[1], raw_f[2])
        incl_lp = grf_inclination_deg(lp_f[0], lp_f[1], lp_f[2])
        lines = [
            f"Frame: {frame_idx}",
            *([f"Knee flex: {knee_flex:.4f}°"] if knee_flex is not None else []),
            f"Plate: {best}  Rate: {fps:.0f} Hz",
            "",
            "Raw   |F|: {:.1f} N  GRF: {:.1f}°".format(raw_mag, incl_raw),
            "LP 10 Hz  |F|: {:.1f} N  GRF: {:.1f}°".format(lp_mag, incl_lp),
        ]
        self.force_output.setPlainText("\n".join(lines))

    def _refresh_analysis_at_frame(self, frame_idx: int):
        """Re-run current analysis type at given frame (for frame navigation)."""
        if self.data is None or frame_idx < 0 or frame_idx >= self.data.n_frames:
            return
        try:
            side = "L" if self.side_left_radio.isChecked() else "R"
            analysis_type = self.analysis_type_combo.currentText()

            if "Shock" in analysis_type or "SA" in analysis_type:
                result = analyze_shock_absorption(self.data, frame_idx=frame_idx, side=side)
                report = format_shock_absorption_report(result, data=self.data)
                geom_2d = get_sa_sagittal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, None, None, result
            elif "Movement" in analysis_type or "MS" in analysis_type:
                result = analyze_movement_strategy(self.data, frame_idx=frame_idx, side=side)
                report = format_movement_strategy_report(result, data=self.data)
                geom_2d = get_ms_sagittal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, None, None, result
            elif "Pelvis" in analysis_type or "PS" in analysis_type:
                result = analyze_pelvis_stability(self.data, frame_idx=frame_idx, side=side)
                report = format_pelvis_stability_report(result, data=self.data)
                geom_2d = get_ps_frontal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, result, None, None
            elif "Trunk" in analysis_type or "TS" in analysis_type:
                result = analyze_trunk_stability(self.data, frame_idx=frame_idx, side=side)
                report = format_trunk_stability_report(result, data=self.data)
                geom_2d = get_ts_frontal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, None, result, None
            else:
                result = analyze_sls_limb_stability(
                    self.data,
                    frame_idx=frame_idx,
                    side=side,
                    use_max_knee_frame=False,
                )
                report = format_limb_stability_report(result, data=self.data)
                geom_2d = get_ls_frontal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = result, None, None, None

            self.analysis_output.setPlainText(report)
            self._last_analysis_frame = frame_idx
            self.fpkpa_diagram.update_diagram(geom_2d)
            self._update_force_display(frame_idx)
            self._show_video_frames(frame_idx)
            self.skeleton_viewer.reset()
            self.skeleton_viewer.update_frame(self.data, frame_idx, ls_result=ls_result, ps_result=ps_result, ts_result=ts_result, ms_result=ms_result)
            self.statusBar().showMessage(f"Frame {frame_idx} – {side} leg")
        except Exception as e:
            self.analysis_output.setPlainText(f"Error: {e}")
            self.statusBar().showMessage(f"Error: {e}")

    def _on_frame_changed(self, value: int):
        if self.data is None or self._last_analysis_frame is None:
            return
        self._refresh_analysis_at_frame(value)

    def _on_frame_prev(self):
        if self.data is None:
            return
        v = self.frame_spin.value()
        if v > 0:
            self.frame_spin.setValue(v - 1)

    def _on_frame_next(self):
        if self.data is None:
            return
        v = self.frame_spin.value()
        if v < self.data.n_frames - 1:
            self.frame_spin.setValue(v + 1)

    def _on_frame_play_toggled(self, checked: bool):
        if self.data is None:
            self.frame_play_btn.setChecked(False)
            return
        if not checked:
            self._play_timer.stop()
            self.frame_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            frame_to_refresh = self.frame_spin.value()
            if self._last_analysis_frame is not None:
                QTimer.singleShot(0, lambda: self._refresh_analysis_at_frame(frame_to_refresh))
            return
        self.frame_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        fps = getattr(self.data, "frame_rate", 60) or 60
        speed_text = self.play_speed_combo.currentText().replace("×", "").replace("x", "").replace("X", "").strip()
        try:
            speed = float(speed_text)
        except ValueError:
            speed = 1.0
        # ใช้ interval คงที่ตาม fps (ให้อัปเดตจอ ~30-60 Hz) แล้วข้ามเฟรมตาม speed แทนการยิง timer ถี่ขึ้น
        base_interval = max(16, int(1000.0 / min(fps, 60)))
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.start(base_interval)

    def _on_play_speed_changed(self):
        """อัปเดตความเร็วเล่นทันทีถ้ากำลังเล่นอยู่ (speed ใช้ใน _on_play_timer เท่านั้น)"""
        if self.data is None or not self.frame_play_btn.isChecked():
            return
        fps = getattr(self.data, "frame_rate", 60) or 60
        base_interval = max(16, int(1000.0 / min(fps, 60)))
        self._play_timer.start(base_interval)

    def _on_play_timer(self):
        if self.data is None or not self.frame_play_btn.isChecked():
            self._play_timer.stop()
            self.frame_play_btn.setChecked(False)
            self.frame_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            return
        v = self.frame_spin.value()
        if v >= self.data.n_frames - 1:
            self._play_timer.stop()
            self.frame_play_btn.setChecked(False)
            self.frame_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.statusBar().showMessage("เล่นจบเฟรมสุดท้าย")
            return
        speed_text = self.play_speed_combo.currentText().replace("×", "").replace("x", "").replace("X", "").strip()
        try:
            speed = max(1, int(float(speed_text)))
        except ValueError:
            speed = 1
        next_frame = min(v + speed, self.data.n_frames - 1)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(next_frame)
        self.frame_spin.blockSignals(False)
        self._update_frame_display_only(next_frame)

    def _update_frame_display_only(self, frame_idx: int):
        """อัปเดตเฉพาะ 3D และแรง ระหว่างเล่น – ไม่อัปเดตวิดีโอเพื่อไม่ให้กระตุก (วิดีโอจะอัปเดตเมื่อกดหยุด)."""
        if self.data is None or frame_idx < 0 or frame_idx >= self.data.n_frames:
            return
        self._update_force_display(frame_idx)
        # ไม่เรียก _show_video_frames ตอนเล่น เพื่อลดการ decode วิดีโอ 2 มุมทุกเฟรม → เล่นลื่น; วิดีโอจะโชว์เฟรมล่าสุดเมื่อกดหยุด
        self.skeleton_viewer.update_frame(
            self.data, frame_idx,
            ls_result=None, ps_result=None, ts_result=None, ms_result=None,
        )

    def run_sls_analysis(self):
        if self.data is None:
            return
        try:
            side = "L" if self.side_left_radio.isChecked() else "R"
            is_hop = self.test_type_combo.currentText() == "Single Leg Hop"
            frame_idx = find_frame_max_knee_flexion(self.data, side, restrict_to_on_plate=is_hop)
            analysis_type = self.analysis_type_combo.currentText()

            if "Shock" in analysis_type or "(SA)" in analysis_type:
                result = analyze_shock_absorption(self.data, frame_idx=frame_idx, side=side)
                report = format_shock_absorption_report(result, data=self.data)
                geom_2d = get_sa_sagittal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, None, None, result
            elif "Movement" in analysis_type or "MS" in analysis_type:
                result = analyze_movement_strategy(self.data, frame_idx=frame_idx, side=side)
                report = format_movement_strategy_report(result, data=self.data)
                geom_2d = get_ms_sagittal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, None, None, result
            elif "Pelvis" in analysis_type or "PS" in analysis_type:
                result = analyze_pelvis_stability(self.data, frame_idx=frame_idx, side=side)
                report = format_pelvis_stability_report(result, data=self.data)
                geom_2d = get_ps_frontal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, result, None, None
            elif "Trunk" in analysis_type or "TS" in analysis_type:
                result = analyze_trunk_stability(self.data, frame_idx=frame_idx, side=side)
                report = format_trunk_stability_report(result, data=self.data)
                geom_2d = get_ts_frontal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = None, None, result, None
            else:
                result = analyze_sls_limb_stability(
                    self.data,
                    frame_idx=frame_idx,
                    side=side,
                    use_max_knee_frame=False,
                )
                report = format_limb_stability_report(result, data=self.data)
                geom_2d = get_ls_frontal_2d_geometry(self.data, result)
                ls_result, ps_result, ts_result, ms_result = result, None, None, None

            self.analysis_output.setPlainText(report)
            self._last_analysis_frame = frame_idx
            self.frame_spin.setMaximum(max(0, self.data.n_frames - 1))
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame_idx)
            self.frame_spin.blockSignals(False)
            self.frame_spin.setEnabled(True)
            self.frame_prev_btn.setEnabled(True)
            self.frame_next_btn.setEnabled(True)
            self.frame_play_btn.setEnabled(True)
            self.fpkpa_diagram.update_diagram(geom_2d)
            self._update_force_display(frame_idx)
            self._show_video_frames(frame_idx)
            self.skeleton_viewer.reset()
            self.skeleton_viewer.update_frame(self.data, frame_idx, ls_result=ls_result, ps_result=ps_result, ts_result=ts_result, ms_result=ms_result)
            test_label = "Hop" if is_hop else "Squat"
            no_fp = is_hop and getattr(self.data, "force_forces", None) is None
            suffix = " (ไม่มีข้อมูล force plate – ใช้ทุกเฟรม)" if no_fp else ""
            self.statusBar().showMessage(f"{test_label} complete – {side} leg, frame {frame_idx}{suffix}")
        except Exception as e:
            self.analysis_output.setPlainText(f"Error: {e}")
            self.force_output.setPlainText(f"Error: {e}")
            self.statusBar().showMessage(f"Analysis error: {e}")
