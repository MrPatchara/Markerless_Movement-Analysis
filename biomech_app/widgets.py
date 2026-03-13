"""
Widget หลัก: 2D diagram, 3D viewer, video label, resize debounce
"""
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QEvent, QSize
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtGui import QResizeEvent, QPainter, QPen, QBrush, QColor, QWheelEvent, QMouseEvent, QFont, QPolygonF

try:
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from biomech_app.theme import COLORS
from biomech.c3d_loader import C3DData
from biomech.sls_analysis import get_ls_overlay_geometry, get_ms_overlay_geometry, get_ps_overlay_geometry, get_ts_overlay_geometry


# --- Resize debounce ---

class ResizeDebounceWidget(QWidget):
    """ห่อ 3D viewer เพื่อ debounce resize – ลด UI freeze ตอนขยายหน้าต่าง"""

    def __init__(self, child: QWidget, parent=None):
        super().__init__(parent)
        self._child = child
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_resize_done)
        self._pending_size = QSize()
        child.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(child)

    def eventFilter(self, obj, event):
        if obj is self._child and event.type() == QEvent.Resize:
            self._pending_size = event.size()
            self._timer.stop()
            self._timer.start(150)
            return True
        return super().eventFilter(obj, event)

    def _on_resize_done(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and self._pending_size.isValid() and self._pending_size.width() > 10 and self._pending_size.height() > 10:
            app.postEvent(self._child, QResizeEvent(self._pending_size, self._child.size()))


# --- Video label ---

class ClickableVideoLabel(QLabel):
    """Label แสดงวิดีโอ และส่งพิกัดเมื่อคลิก (สำหรับวางจุด 2D)"""
    from PySide6.QtCore import Signal
    point_clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._placement_mode = False

    def set_placement_mode(self, on: bool):
        self._placement_mode = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if self._placement_mode and event.button() == Qt.LeftButton:
            p = event.position()
            self.point_clicked.emit(p.x(), p.y())
        super().mousePressEvent(event)


# --- 2D FPKPA diagram ---

class FPKPADiagramWidget(QWidget):
    """แผนภาพ 2D frontal plane สำหรับ FPKPA – ใช้ geometry เดียวกับที่คำนวณ"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._geom = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start: Optional[tuple] = None
        self.setMinimumSize(220, 250)
        self.setStyleSheet(f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['bg_elevated']}; border-radius: 6px;")
        self.setToolTip("2D Frontal Plane – Scroll: zoom | ลากเมาส์ซ้าย: เลื่อน")

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(4.0, self._zoom * 1.15)
        else:
            self._zoom = max(0.3, self._zoom / 1.15)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (event.position().x(), event.position().y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            dx = event.position().x() - self._drag_start[0]
            dy = event.position().y() - self._drag_start[1]
            self._pan_x += dx
            self._pan_y += dy
            self._drag_start = (event.position().x(), event.position().y())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
        super().mouseReleaseEvent(event)

    def update_diagram(self, geom):
        """อัปเดต geometry จาก get_ls_frontal_2d_geometry"""
        self._geom = geom
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._geom is None:
            return
        analysis_type = self._geom.get("analysis_type", "LS")
        skel = self._geom.get("skeleton_2d")
        conns = self._geom.get("connections") or []

        if analysis_type in ("PS", "TS"):
            lasis = self._geom.get("lasis")
            rasis = self._geom.get("rasis")
            if lasis is None or rasis is None:
                return
            pts_list = [lasis, rasis]
        elif analysis_type == "MS":
            lm = self._geom.get("lm")
            lfc = self._geom.get("lfc")
            gt = self._geom.get("gt")
            sjc = self._geom.get("sjc")
            if lm is None or lfc is None or gt is None or sjc is None:
                return
            pts_list = [lm, lfc, gt, sjc]
        elif analysis_type == "SA":
            lm = self._geom.get("lm")
            lfc = self._geom.get("lfc")
            gt = self._geom.get("gt")
            if lm is None or lfc is None or gt is None:
                return
            pts_list = [lm, lfc, gt]
        else:
            asis = self._geom.get("asis")
            kjc = self._geom.get("kjc")
            ajc = self._geom.get("ajc")
            if asis is None or kjc is None or ajc is None:
                return
            arc = self._geom.get("arc_points")
            pts_list = [asis, kjc, ajc]
            for k in ("lfc",):
                p = self._geom.get(k)
                if p is not None and np.isfinite(p).all():
                    pts_list.append(p)
            if arc is not None:
                pts_list.append(arc)
        if skel is not None and len(skel) > 0:
            valid = np.isfinite(skel).all(axis=1)
            if np.any(valid):
                pts_list.append(skel[valid])
        if analysis_type in ("PS", "TS"):
            h_ref = self._geom.get("horizontal_ref")
            if h_ref is not None:
                pts_list.extend([h_ref[0], h_ref[1]])
        elif analysis_type == "MS":
            pass  # pts_list already has lm, lfc, gt, sjc
        elif analysis_type == "SA":
            pass  # pts_list already has lm, lfc, gt
        pts = np.vstack(pts_list) if pts_list else np.array([[0, 0], [1, 1]])
        mn = np.nanmin(pts, axis=0)
        mx = np.nanmax(pts, axis=0)
        rg = mx - mn
        if np.any(rg < 1e-6):
            rg[rg < 1e-6] = 1.0
        margin = 24
        w, h = self.width() - 2 * margin, self.height() - 2 * margin
        scale = min(w / rg[0], h / rg[1]) * 1.25 * getattr(self, "_zoom", 1.0)
        cx = (mn[0] + mx[0]) / 2
        cy = (mn[1] + mx[1]) / 2
        pan_x = getattr(self, "_pan_x", 0.0)
        pan_y = getattr(self, "_pan_y", 0.0)
        flip_ml = self._geom.get("flip_ml", False)
        sagittal_y_down = self._geom.get("sagittal_y_down", False)  # MS: larger y = bottom (foot down)

        def to_qt(x, y):
            dx = (cx - x) * scale if flip_ml else (x - cx) * scale
            sx = margin + w / 2 + dx + pan_x
            if sagittal_y_down:
                sy = margin + h / 2 + (y - cy) * scale + pan_y
            else:
                sy = margin + h / 2 - (y - cy) * scale + pan_y
            return sx, sy

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if skel is not None and len(conns) > 0:
            skel_pen = QPen(QColor("#4A6A8A"), 1)
            skel_pt_pen = QPen(QColor("#6B8CBF"), 1)
            painter.setPen(skel_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for a, b in conns:
                if a < len(skel) and b < len(skel) and np.isfinite(skel[a]).all() and np.isfinite(skel[b]).all():
                    p1 = to_qt(skel[a, 0], skel[a, 1])
                    p2 = to_qt(skel[b, 0], skel[b, 1])
                    painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            painter.setPen(skel_pt_pen)
            painter.setBrush(QBrush(QColor("#5A7A9A"), Qt.BrushStyle.SolidPattern))
            for i in range(len(skel)):
                if np.isfinite(skel[i]).all():
                    px, py = to_qt(skel[i, 0], skel[i, 1])
                    painter.drawEllipse(int(px - 1), int(py - 1), 2, 2)
        thigh_pen = QPen(QColor("#FF8C42"), 1)
        shank_pen = QPen(QColor("#00C8E0"), 1)
        pt_pen = QPen(QColor("#00D4AA"), 1)
        pelvis_pen = QPen(QColor("#E040A0"), 2)
        horiz_pen = QPen(QColor("#808080"), 1)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if analysis_type in ("PS", "TS"):
            painter.setPen(QPen(QColor(COLORS["text"])))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(8, 16, "Frontal view")
            lasis = self._geom.get("lasis")
            rasis = self._geom.get("rasis")
            pelvis_line = self._geom.get("pelvis_line")
            horiz_ref = self._geom.get("horizontal_ref")
            if analysis_type == "TS":
                point_labels = ("midline pelvis", "clavicular")
                angle_prefix = "TA"
            else:
                point_labels = ("LASIS", "RASIS")
                angle_prefix = "PA"
            if pelvis_line:
                p1, p2 = pelvis_line
                a1 = to_qt(p1[0], p1[1])
                a2 = to_qt(p2[0], p2[1])
                painter.setPen(pelvis_pen)
                painter.drawLine(int(a1[0]), int(a1[1]), int(a2[0]), int(a2[1]))
            if horiz_ref:
                h1, h2 = horiz_ref
                b1 = to_qt(h1[0], h1[1])
                b2 = to_qt(h2[0], h2[1])
                painter.setPen(horiz_pen)
                painter.drawLine(int(b1[0]), int(b1[1]), int(b2[0]), int(b2[1]))
                if analysis_type == "TS":
                    vert_label = self._geom.get("vertical_ref_label")
                    if vert_label:
                        mid_h = ((h1[0] + h2[0]) / 2, (h1[1] + h2[1]) / 2)
                        top_qt = to_qt(mid_h[0], max(h1[1], h2[1]))
                        painter.setPen(QPen(QColor("#808080")))
                        painter.setFont(QFont("Segoe UI", 7))
                        painter.drawText(int(top_qt[0]) + 4, int(top_qt[1]) - 2, vert_label)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            for name, p in zip(point_labels, (lasis, rasis)):
                if p is not None and np.isfinite(p).all():
                    painter.setPen(pt_pen)
                    painter.setBrush(QBrush(QColor("#00D4AA"), Qt.BrushStyle.SolidPattern))
                    px, py = to_qt(p[0], p[1])
                    painter.drawEllipse(int(px - 3), int(py - 3), 6, 6)
                    painter.setPen(QPen(QColor(COLORS["text"])))
                    painter.drawText(int(px + 6), int(py + 3), name)
            ang = self._geom.get("angle_deg")
            if ang is not None:
                mid = (lasis + rasis) / 2 if lasis is not None and rasis is not None else lasis
                if mid is not None:
                    pm = to_qt(mid[0], mid[1])
                    painter.setPen(QPen(QColor("#FFD700")))
                    painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                    painter.drawText(int(pm[0]) - 24, int(pm[1]) - 18, f"{angle_prefix}: {ang:+.2f}\u00B0")
        elif analysis_type == "MS":
            lm = self._geom.get("lm")
            lfc = self._geom.get("lfc")
            gt = self._geom.get("gt")
            sjc = self._geom.get("sjc")
            line_shank = self._geom.get("line_shank")
            line_thigh = self._geom.get("line_thigh")
            line_trunk = self._geom.get("line_trunk")
            alpha_deg = self._geom.get("alpha_deg")
            beta_deg = self._geom.get("beta_deg")
            painter.setPen(QPen(QColor(COLORS["text"])))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(8, 16, "Sagittal view")
            if line_shank:
                s1, s2 = line_shank
                p1 = to_qt(s1[0], s1[1])
                p2 = to_qt(s2[0], s2[1])
                painter.setPen(shank_pen)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                mid = ((s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2)
                pm = to_qt(mid[0], mid[1])
                painter.setPen(QPen(QColor("#00C8E0")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(int(pm[0]) + 4, int(pm[1]) - 2, "Shank")
            if line_thigh:
                t1, t2 = line_thigh
                p1 = to_qt(t1[0], t1[1])
                p2 = to_qt(t2[0], t2[1])
                painter.setPen(thigh_pen)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                mid = ((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2)
                pm = to_qt(mid[0], mid[1])
                painter.setPen(QPen(QColor("#FF8C42")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(int(pm[0]) + 4, int(pm[1]) - 2, "Thigh")
            if line_trunk:
                r1, r2 = line_trunk
                p1 = to_qt(r1[0], r1[1])
                p2 = to_qt(r2[0], r2[1])
                painter.setPen(pelvis_pen)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                mid = ((r1[0] + r2[0]) / 2, (r1[1] + r2[1]) / 2)
                pm = to_qt(mid[0], mid[1])
                painter.setPen(QPen(QColor("#E040A0")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(int(pm[0]) + 4, int(pm[1]) - 2, "Trunk")
            for name, p in [("LM", lm), ("LFC", lfc), ("GT", gt), ("SJC", sjc)]:
                if p is not None and np.isfinite(p).all():
                    painter.setPen(pt_pen)
                    painter.setBrush(QBrush(QColor("#00D4AA"), Qt.BrushStyle.SolidPattern))
                    px, py = to_qt(p[0], p[1])
                    painter.drawEllipse(int(px - 3), int(py - 3), 6, 6)
                    painter.setPen(QPen(QColor(COLORS["text"])))
                    painter.drawText(int(px + 6), int(py + 3), name)
            if lfc is not None and np.isfinite(lfc).all() and alpha_deg is not None:
                pk = to_qt(lfc[0], lfc[1])
                painter.setPen(QPen(QColor("#FFD700")))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(int(pk[0]) - 20, int(pk[1]) - 22, f"\u03B1: {alpha_deg:.2f}\u00B0")
            if gt is not None and np.isfinite(gt).all() and beta_deg is not None:
                pg = to_qt(gt[0], gt[1])
                painter.setPen(QPen(QColor("#FFD700")))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(int(pg[0]) - 20, int(pg[1]) - 22, f"\u03B2: {beta_deg:.2f}\u00B0")
        elif analysis_type == "SA":
            lm = self._geom.get("lm")
            lfc = self._geom.get("lfc")
            gt = self._geom.get("gt")
            line_shank = self._geom.get("line_shank")
            line_thigh = self._geom.get("line_thigh")
            alpha_deg = self._geom.get("alpha_deg")
            painter.setPen(QPen(QColor(COLORS["text"])))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(8, 16, "Sagittal (Knee flexion \u03B1)")
            if line_shank:
                s1, s2 = line_shank
                p1 = to_qt(s1[0], s1[1])
                p2 = to_qt(s2[0], s2[1])
                painter.setPen(shank_pen)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                mid = ((s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2)
                pm = to_qt(mid[0], mid[1])
                painter.setPen(QPen(QColor("#00C8E0")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(int(pm[0]) + 4, int(pm[1]) - 2, "Shank")
            if line_thigh:
                t1, t2 = line_thigh
                p1 = to_qt(t1[0], t1[1])
                p2 = to_qt(t2[0], t2[1])
                painter.setPen(thigh_pen)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                mid = ((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2)
                pm = to_qt(mid[0], mid[1])
                painter.setPen(QPen(QColor("#FF8C42")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(int(pm[0]) + 4, int(pm[1]) - 2, "Thigh")
            for name, p in [("LM", lm), ("LFC", lfc), ("GT", gt)]:
                if p is not None and np.isfinite(p).all():
                    painter.setPen(pt_pen)
                    painter.setBrush(QBrush(QColor("#00D4AA"), Qt.BrushStyle.SolidPattern))
                    px, py = to_qt(p[0], p[1])
                    painter.drawEllipse(int(px - 3), int(py - 3), 6, 6)
                    painter.setPen(QPen(QColor(COLORS["text"])))
                    painter.drawText(int(px + 6), int(py + 3), name)
            if lfc is not None and np.isfinite(lfc).all() and alpha_deg is not None:
                pk = to_qt(lfc[0], lfc[1])
                painter.setPen(QPen(QColor("#FFD700")))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(int(pk[0]) - 20, int(pk[1]) - 22, f"\u03B1: {alpha_deg:.2f}\u00B0")
        else:
            painter.setPen(QPen(QColor(COLORS["text"])))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(8, 16, "Frontal view")
            asis = self._geom.get("asis")
            kjc = self._geom.get("kjc")
            ajc = self._geom.get("ajc")
            thigh_ext = self._geom.get("line_thigh_ext")
            shank_ext = self._geom.get("line_shank_ext")
            if thigh_ext and shank_ext:
                t1, t2 = thigh_ext
                s1, s2 = shank_ext
                pt1 = to_qt(t1[0], t1[1])
                pt2 = to_qt(t2[0], t2[1])
                ps1 = to_qt(s1[0], s1[1])
                ps2 = to_qt(s2[0], s2[1])
                painter.setPen(thigh_pen)
                painter.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
                painter.setPen(shank_pen)
                painter.drawLine(int(ps1[0]), int(ps1[1]), int(ps2[0]), int(ps2[1]))
            else:
                p_asis = to_qt(asis[0], asis[1])
                p_kjc = to_qt(kjc[0], kjc[1])
                p_ajc = to_qt(ajc[0], ajc[1])
                painter.setPen(thigh_pen)
                painter.drawLine(int(p_asis[0]), int(p_asis[1]), int(p_kjc[0]), int(p_kjc[1]))
                painter.setPen(shank_pen)
                painter.drawLine(int(p_kjc[0]), int(p_kjc[1]), int(p_ajc[0]), int(p_ajc[1]))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            for name, p in [("ASIS", asis), ("KJC", kjc), ("AJC", ajc)]:
                painter.setPen(pt_pen)
                painter.setBrush(QBrush(QColor("#00D4AA"), Qt.BrushStyle.SolidPattern))
                px, py = to_qt(p[0], p[1])
                painter.drawEllipse(int(px - 3), int(py - 3), 6, 6)
                painter.setPen(QPen(QColor(COLORS["text"])))
                painter.drawText(int(px + 6), int(py + 3), name)
            lfc = self._geom.get("lfc")
            if lfc is not None and np.isfinite(lfc).all():
                painter.setPen(QPen(QColor("#8888FF"), 1))
                painter.setBrush(QBrush(QColor("#8888FF"), Qt.BrushStyle.SolidPattern))
                lx, ly = to_qt(lfc[0], lfc[1])
                painter.drawEllipse(int(lx - 2), int(ly - 2), 4, 4)
            ang = self._geom.get("angle_deg")
            interp = self._geom.get("angle_interpretation", "")
            if ang is not None:
                pk = to_qt(kjc[0], kjc[1])
                painter.setPen(QPen(QColor("#FFD700")))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                display_ang = -ang if interp == "varus" else ang
                lbl = f"{display_ang:+.4f}\u00B0" + (f" ({interp})" if interp else "")
                painter.drawText(int(pk[0]) - 24, int(pk[1]) - 18, lbl)


# --- 3D Skeleton viewer ---

class SkeletonViewerWidget(QtInteractor if PYVISTA_AVAILABLE else QFrame):
    """แสดง skeleton 3D ของ 1 เฟรม – ปรับแกนให้ตรงกับ C3D อัตโนมัติ"""

    def __init__(self, parent=None):
        if not PYVISTA_AVAILABLE:
            super().__init__(parent)
            self.setMinimumSize(320, 320)
            lbl = QLabel("Install pyvista, pyvistaqt for 3D view")
            lbl.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout(self)
            layout.addWidget(lbl)
            return
        super().__init__(parent)
        self.set_background("#0D1B2A")
        self.setMinimumSize(320, 320)
        self._markers_actor = None
        self._bones_actor = None
        self._lines_mesh = None
        self._n_lines = 0
        self._ref_center = None
        self._ref_scale = 1.0
        self._axis_perm = None
        self._axis_flip = 1
        self._basis = None  # 3x3 world basis: rows = [ML, Vertical, AP]
        self._force_plate_actors = []
        self._ls_overlay_actors = []

    def _remove_ls_overlay(self):
        for actor in self._ls_overlay_actors:
            try:
                self.remove_actor(actor)
            except Exception:
                pass
        self._ls_overlay_actors.clear()

    def _add_force_plates(self, data: C3DData):
        fp_list = getattr(data, "force_plates", None) or []
        if not fp_list:
            return
        fp_color = "#4A90A4"
        for corners in fp_list:
            pts = np.asarray(corners, dtype=np.float64)
            if pts.shape[0] < 4:
                continue
            pts = pts[:4]
            pts_t = self._transform(pts)
            if np.any(~np.isfinite(pts_t)):
                continue
            faces = np.array([4, 0, 1, 2, 3], dtype=np.int32)
            mesh = pv.PolyData(pts_t, faces=faces)
            actor = self.add_mesh(mesh, color=fp_color, opacity=0.5, smooth_shading=True)
            self._force_plate_actors.append(actor)

    def _remove_force_plates(self):
        for actor in self._force_plate_actors:
            try:
                self.remove_actor(actor)
            except Exception:
                pass
        self._force_plate_actors.clear()

    def _add_ls_overlay(self, data: C3DData, ls_result):
        try:
            geom = get_ls_overlay_geometry(data, ls_result)
        except Exception:
            return
        fpkpa_color = "#FFB347"
        arc_color = "#FF6B6B"
        label_color = "#00D4AA"
        if geom.get("line_thigh"):
            p1, p2 = geom["line_thigh"]
            pts_t = self._transform(np.array([p1, p2]))
            if np.isfinite(pts_t).all():
                line = pv.Line(pts_t[0], pts_t[1])
                actor = self.add_mesh(line, color=fpkpa_color, line_width=4)
                self._ls_overlay_actors.append(actor)
        if geom.get("line_shank"):
            p1, p2 = geom["line_shank"]
            pts_t = self._transform(np.array([p1, p2]))
            if np.isfinite(pts_t).all():
                line = pv.Line(pts_t[0], pts_t[1])
                actor = self.add_mesh(line, color=fpkpa_color, line_width=4)
                self._ls_overlay_actors.append(actor)
        if geom.get("arc_points") is not None:
            arc = geom["arc_points"]
            if len(arc) >= 2:
                pts_t = self._transform(arc)
                if np.isfinite(pts_t).all():
                    poly = pv.PolyData(pts_t, lines=np.concatenate([[len(pts_t)] + list(range(len(pts_t)))]))
                    actor = self.add_mesh(poly, color=arc_color, line_width=3)
                    self._ls_overlay_actors.append(actor)
        for key, color in [("kjc", label_color), ("lfc", "#8888FF")]:
            pt = geom.get(key)
            if pt is not None and np.isfinite(pt).all():
                pt_t = self._transform(pt.reshape(1, -1)).ravel()
                if np.isfinite(pt_t).all():
                    sph = pv.Sphere(radius=0.025, center=pt_t)
                    actor = self.add_mesh(sph, color=color, opacity=0.9)
                    self._ls_overlay_actors.append(actor)

    def _add_ps_overlay(self, data: C3DData, ps_result):
        """Pelvis Stability: เส้น LASIS–RASIS + แกนแนวนอน (horizontal ref) ที่ใช้คำนวณ PA."""
        try:
            geom = get_ps_overlay_geometry(data, ps_result)
        except Exception:
            return
        line_color = "#E040A0"
        axis_color = "#9E9E9E"
        pt_color = "#00D4AA"
        if geom.get("line_pelvis"):
            p1, p2 = geom["line_pelvis"]
            pts_t = self._transform(np.array([p1, p2]))
            if np.isfinite(pts_t).all():
                line = pv.Line(pts_t[0], pts_t[1])
                self._ls_overlay_actors.append(self.add_mesh(line, color=line_color, line_width=4))
        if geom.get("horizontal_ref_3d"):
            p1, p2 = geom["horizontal_ref_3d"]
            pts_t = self._transform(np.array([p1, p2]))
            if np.isfinite(pts_t).all():
                line = pv.Line(pts_t[0], pts_t[1])
                self._ls_overlay_actors.append(self.add_mesh(line, color=axis_color, line_width=2))
        for key in ("lasis", "rasis"):
            pt = geom.get(key)
            if pt is not None and np.isfinite(pt).all():
                pt_t = self._transform(pt.reshape(1, -1)).ravel()
                if np.isfinite(pt_t).all():
                    sph = pv.Sphere(radius=0.025, center=pt_t)
                    self._ls_overlay_actors.append(self.add_mesh(sph, color=pt_color, opacity=0.9))

    def _add_ts_overlay(self, data: C3DData, ts_result):
        """Trunk Stability: เส้น trunk (clavicular–midline pelvis) + แกนแนวตั้ง (vertical/gravity ref)."""
        try:
            geom = get_ts_overlay_geometry(data, ts_result)
        except Exception:
            return
        line_color = "#E040A0"
        axis_color = "#9E9E9E"
        pt_color = "#00D4AA"
        if geom.get("line_trunk"):
            p1, p2 = geom["line_trunk"]
            pts_t = self._transform(np.array([p1, p2]))
            if np.isfinite(pts_t).all():
                line = pv.Line(pts_t[0], pts_t[1])
                self._ls_overlay_actors.append(self.add_mesh(line, color=line_color, line_width=4))
        if geom.get("vertical_ref_3d"):
            p1, p2 = geom["vertical_ref_3d"]
            pts_t = self._transform(np.array([p1, p2]))
            if np.isfinite(pts_t).all():
                line = pv.Line(pts_t[0], pts_t[1])
                self._ls_overlay_actors.append(self.add_mesh(line, color=axis_color, line_width=2))
        for key in ("midline_pelvis", "clavicular"):
            pt = geom.get(key)
            if pt is not None and np.isfinite(pt).all():
                pt_t = self._transform(pt.reshape(1, -1)).ravel()
                if np.isfinite(pt_t).all():
                    sph = pv.Sphere(radius=0.025, center=pt_t)
                    self._ls_overlay_actors.append(self.add_mesh(sph, color=pt_color, opacity=0.9))

    def _add_ms_overlay(self, data: C3DData, ms_result):
        """Movement Strategy: เส้น LM–LFC (shank), LFC–GT (thigh), GT–SJC (trunk) และจุด LM, LFC, GT, SJC"""
        try:
            geom = get_ms_overlay_geometry(data, ms_result)
        except Exception:
            return
        shank_color = "#00C8E0"
        thigh_color = "#FF8C42"
        trunk_color = "#E040A0"
        pt_color = "#00D4AA"
        for key, color in [("line_shank", shank_color), ("line_thigh", thigh_color), ("line_trunk", trunk_color)]:
            line = geom.get(key)
            if line:
                p1, p2 = line
                pts_t = self._transform(np.array([p1, p2]))
                if np.isfinite(pts_t).all():
                    line_mesh = pv.Line(pts_t[0], pts_t[1])
                    self._ls_overlay_actors.append(self.add_mesh(line_mesh, color=color, line_width=4))
        for key in ("lm", "lfc", "gt", "sjc"):
            pt = geom.get(key)
            if pt is not None and np.isfinite(pt).all():
                pt_t = self._transform(pt.reshape(1, -1)).ravel()
                if np.isfinite(pt_t).all():
                    sph = pv.Sphere(radius=0.025, center=pt_t)
                    self._ls_overlay_actors.append(self.add_mesh(sph, color=pt_color, opacity=0.9))

    def _compute_orientation(self, data: C3DData):
        """กำหนดแกน world สำหรับ 3D viewer ให้สอดคล้องกับ force plate เมื่อมี (รองรับทุก C3D)."""
        self._basis = None
        self._axis_perm = None
        self._axis_flip = 1
        all_pts = data.points.reshape(-1, 3)
        valid = np.isfinite(all_pts).all(axis=1)
        if not np.any(valid):
            self._axis_perm = [0, 1, 2]
            self._axis_flip = 1
            return

        pts_valid = all_pts[valid]

        # 1) ถ้ามี force plate: ใช้ geometry ของแผ่นกำหนด vertical/ML (เหมือนใน sls_analysis)
        fp_list = getattr(data, "force_plates", None) or []
        if fp_list:
            crn = np.asarray(fp_list[0], dtype=float)
            if crn.shape == (3, 4):
                crn = crn.T
            crn = crn[:4]
            # normal ของแผ่น = แกนตั้ง (ยังไม่รู้ว่าชี้ขึ้นหรือลง)
            e1 = crn[1] - crn[0]
            e2 = crn[3] - crn[0]
            vertical = np.cross(e1, e2)
            n_vert = np.linalg.norm(vertical)
            if n_vert > 1e-8:
                vertical = vertical / n_vert
                # ML: โปรเจกต์ขอบแผ่นลงบนระนาบแนวราบ
                e1_h = e1 - np.dot(e1, vertical) * vertical
                n_ml = np.linalg.norm(e1_h)
                if n_ml < 1e-8:
                    e2_h = e2 - np.dot(e2, vertical) * vertical
                    n_ml2 = np.linalg.norm(e2_h)
                    ml = e2_h / n_ml2 if n_ml2 > 1e-8 else None
                else:
                    ml = e1_h / n_ml if n_ml > 1e-8 else None
                if ml is not None:
                    plate_center = np.nanmean(crn, axis=0)
                    centroid = np.nanmean(pts_valid, axis=0)
                    dot = float(np.dot(centroid - plate_center, vertical))
                    v_flip = 1.0 if dot > 0 else -1.0
                    e_y = v_flip * vertical
                    e_x = ml
                    e_z = np.cross(e_x, e_y)
                    n_z = np.linalg.norm(e_z)
                    if n_z > 1e-8:
                        e_z = e_z / n_z
                        self._basis = np.vstack([e_x, e_y, e_z])
                        return

        # 2) ถ้าไม่มี force plate หรือคำนวณไม่สำเร็จ: fallback heuristic เดิมตาม range ของแกน
        extents = np.ptp(pts_valid, axis=0)
        up_axis = int(np.argmax(extents))
        others = [i for i in range(3) if i != up_axis]
        self._axis_perm = [others[0], up_axis, others[1]]
        up_vals = pts_valid[:, up_axis]
        mid = np.median(up_vals)
        above = np.nanmean(up_vals[up_vals > mid]) if np.any(up_vals > mid) else mid
        below = np.nanmean(up_vals[up_vals <= mid]) if np.any(up_vals <= mid) else mid
        self._axis_flip = 1 if above > below else -1

    def _transform(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 1:
            pts = points.reshape(1, -1)
        else:
            pts = points
        if self._basis is not None:
            # ใช้ basis จาก force plate: [ML, Vertical, AP]
            out = pts @ self._basis.T
        elif self._axis_perm is not None:
            out = np.column_stack([
                pts[:, self._axis_perm[0]],
                self._axis_flip * pts[:, self._axis_perm[1]],
                pts[:, self._axis_perm[2]],
            ])
        else:
            out = pts.copy()
        if self._ref_center is None or self._ref_scale is None:
            c = np.nanmean(out, axis=0)
            s = np.nanmax(np.abs(out - c)) or 1.0
            return (out - c) / s * 0.9
        return (out - self._ref_center) / self._ref_scale * 0.9

    def update_frame(self, data: C3DData, frame_idx: int, ls_result=None, ps_result=None, ts_result=None, ms_result=None):
        if not PYVISTA_AVAILABLE or data is None:
            return
        self._remove_ls_overlay()
        points = data.get_frame(frame_idx).copy()
        connections = data.connections or []
        valid = np.isfinite(points).all(axis=1)
        if not valid.any():
            return
        if self._axis_perm is None:
            self._compute_orientation(data)
        if self._ref_center is None:
            all_pts = data.points.reshape(-1, 3)
            valid_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
            if len(valid_pts) > 0:
                oriented = self._transform(valid_pts)
                self._ref_center = np.nanmean(oriented, axis=0)
                self._ref_scale = max(np.nanmax(np.abs(oriented - self._ref_center)), 1e-6)
        pts = self._transform(points)
        marker_color, bone_color = "#FFFFFF", "#6B8CBF"
        valid_mask = np.isfinite(pts).all(axis=1)
        marker_pts = pts[valid_mask]
        if len(marker_pts) > 0:
            cloud = pv.PolyData(marker_pts)
            radius = 0.018
            spheres = cloud.glyph(scale=False, orient=False, geom=pv.Sphere(radius=radius))
            if self._markers_actor is not None:
                try:
                    mapper = getattr(self._markers_actor, "mapper", None) or self._markers_actor.GetMapper()
                    mapper.SetInputData(spheres)
                    mapper.Update()
                except Exception:
                    try:
                        self.remove_actor(self._markers_actor)
                    except Exception:
                        pass
                    self._markers_actor = self.add_mesh(spheres, color=marker_color, smooth_shading=True)
            else:
                self._markers_actor = self.add_mesh(spheres, color=marker_color, smooth_shading=True)
        elif self._markers_actor is not None:
            try:
                self.remove_actor(self._markers_actor)
            except Exception:
                pass
            self._markers_actor = None
        line_pts, line_cells = [], []
        for start, end in connections:
            if start < len(pts) and end < len(pts):
                p1, p2 = pts[start], pts[end]
                if np.isfinite(p1).all() and np.isfinite(p2).all():
                    line_pts.extend([p1, p2])
                    line_cells.extend([2, len(line_pts) - 2, len(line_pts) - 1])
        if line_pts:
            arr = np.array(line_pts)
            cells = np.array(line_cells, dtype=np.int32)
            if self._bones_actor and self._lines_mesh and len(line_pts) == self._n_lines * 2:
                self._lines_mesh.points = arr.astype(np.float32)
                self._lines_mesh.Modified()
            else:
                if self._bones_actor:
                    try:
                        self.remove_actor(self._bones_actor)
                    except Exception:
                        pass
                self._n_lines = len(line_pts) // 2
                self._lines_mesh = pv.PolyData(arr, lines=cells)
                self._bones_actor = self.add_mesh(self._lines_mesh, color=bone_color, line_width=2)
        self._remove_force_plates()
        if getattr(data, "force_plates", None):
            self._add_force_plates(data)
        if ps_result is not None:
            self._add_ps_overlay(data, ps_result)
        elif ts_result is not None:
            self._add_ts_overlay(data, ts_result)
        elif ms_result is not None:
            self._add_ms_overlay(data, ms_result)
        elif ls_result is not None:
            self._add_ls_overlay(data, ls_result)
        # ตั้งมุมมองเริ่มต้นให้เป็น frontal view (มองจากด้านหน้า ML–Vertical plane)
        if ps_result is not None or ts_result is not None or ms_result is not None or ls_result is not None:
            try:
                self.view_xy()
                self.set_viewup([0, 1, 0])
            except Exception:
                pass
        self.render()

    def reset(self):
        if not PYVISTA_AVAILABLE:
            return
        if self._markers_actor is not None:
            try:
                self.remove_actor(self._markers_actor)
            except Exception:
                pass
            self._markers_actor = None
        if self._bones_actor is not None:
            try:
                self.remove_actor(self._bones_actor)
            except Exception:
                pass
            self._bones_actor = None
            self._lines_mesh = None
        self._n_lines = 0
        self._ref_center = None
        self._ref_scale = 1.0
        self._axis_perm = None
        self._axis_flip = 1
        self._remove_force_plates()
        self._remove_ls_overlay()
        self.render()
