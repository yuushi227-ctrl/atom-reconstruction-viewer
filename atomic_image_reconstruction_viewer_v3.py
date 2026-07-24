#!/usr/bin/env python3
"""
Atom Image Viewer - 原子像再構成ビューア (3D-AIR-IMAGE API対応版)
==================================================================
機能:
  1. 3D-AIR-IMAGE API から直接ボリュームデータを取得・表示
  2. XY / XZ / YZ の3平面スライス表示
  3. 2点クリックによる原子間距離測定（ピーク自動検出 + サブピクセル精度）
  4. XYZファイル読み込みによる原子位置オーバーレイ表示
  5. .npy / .npz / .xml_air ファイルからの読み込みにも対応

距離単位: Å（オングストローム）
ピーク探索半径: ユーザー調整可能

3D-AIR-IMAGE APIメソッド（使用しているもの）:
  - api_3d_air_image_qt6.Api()         → インスタンス生成
  - .connect()                          → 接続
  - .window_get_list_wid_and_caption()  → ウィンドウ一覧
  - .window_is_mode_series_get(wid)     → シリーズウィンドウか判定
  - .image_get(wid)                     → 2D画像をNumPy配列で取得
  - .series_get_count(wid)              → シリーズのフレーム数
  - .series_imageindex_set(wid, i)      → シリーズのフレーム切替
"""

import sys
import os
import functools
import numpy as np
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
        QComboBox, QPushButton, QFileDialog, QGroupBox, QStatusBar,
        QMessageBox, QCheckBox, QSplitter, QFrame, QTableWidget,
        QTableWidgetItem, QHeaderView, QToolBar, QAction, QSizePolicy,
        QDialog, QListWidget, QDialogButtonBox, QProgressBar,
        QListWidgetItem, QScrollArea, QColorDialog,
        QRadioButton, QButtonGroup, QTabWidget, QProgressDialog, QShortcut
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QThread, pyqtSlot
    from PyQt5.QtGui import QFont, QColor, QIcon, QKeySequence
except ImportError:
    print("PyQt5が必要です: pip install PyQt5")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
except ImportError:
    print("Matplotlibが必要です: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.ndimage import maximum_filter, gaussian_filter
    from scipy import ndimage
except ImportError:
    print("SciPyが必要です: pip install scipy")
    sys.exit(1)

# =============================================================================
# 3D-AIR-IMAGE API 接続
# =============================================================================
AIR_API_AVAILABLE = False
air_api = None

try:
    import api_3d_air_image_qt6
    AIR_API_AVAILABLE = True
except ImportError:
    print("[INFO] api_3d_air_image_qt6 が見つかりません。"
          "ファイル読み込みモードで動作します。")


def connect_air_api():
    """3D-AIR-IMAGE APIに接続してApiオブジェクトを返す"""
    global air_api
    if not AIR_API_AVAILABLE:
        return None
    if air_api is not None:
        return air_api
    try:
        air_api = api_3d_air_image_qt6.Api()
        air_api.connect()
        return air_api
    except Exception as e:
        print(f"[WARNING] 3D-AIR-IMAGE API接続失敗: {e}")
        return None


# =============================================================================
# XYZファイル読み込み
# =============================================================================
def load_xyz(filepath: str) -> list:
    """XYZ形式読み込み（標準ヘッダーあり・なし両対応）
    標準形式: 行1=原子数 / 行2=コメント / 行3-: 元素 x y z
    非標準形式: 元素 x y z のみの行が並ぶ
    """
    atoms = []
    for enc in ('utf-8-sig', 'utf-8', 'shift_jis', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                raw_lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("ファイルのエンコーディングを判別できませんでした")

    lines = [l for l in raw_lines if l.strip()]
    if not lines:
        return atoms

    # 1行目が整数なら標準XYZ形式 → 2行スキップ
    start = 0
    try:
        int(lines[0].strip())
        start = 2
    except ValueError:
        start = 0  # ヘッダーなしとして全行パース

    for i in range(start, len(lines)):
        parts = lines[i].split()
        if len(parts) >= 4:
            try:
                atoms.append({
                    'element': parts[0],
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                })
            except ValueError:
                continue  # 数値でない行はスキップ
    return atoms


# =============================================================================
# 元素の色 / 半径マッピング
# =============================================================================
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
    'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
    'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
    'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
    'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033',
    'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3',
    'Se': '#FFA100', 'Br': '#A62929', 'Sr': '#00FF00', 'Y': '#94FFFF',
    'Zr': '#94E0E0', 'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Pd': '#006985',
    'Ag': '#C0C0C0', 'Au': '#FFD123', 'Pt': '#D0D0E0', 'Pb': '#575961',
}

ELEMENT_RADII = {
    'H': 0.25, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'Si': 1.17,
    'Al': 1.43, 'Fe': 1.26, 'Cu': 1.28, 'Zn': 1.34, 'Ga': 1.22,
    'Ge': 1.22, 'As': 1.21, 'Ti': 1.47, 'Sr': 2.15, 'Au': 1.44,
    'Ag': 1.45, 'Pt': 1.39, 'default': 0.80,
}


def get_element_color(element):
    return ELEMENT_COLORS.get(element, '#FF69B4')


def get_element_radius(element):
    return ELEMENT_RADII.get(element, ELEMENT_RADII['default'])


# =============================================================================
# ピーク精緻化定数
# =============================================================================
PEAK_REFINE_GAUSSIAN = 0
PEAK_REFINE_CENTROID = 1
PEAK_REFINE_PARABOLA = 2
PEAK_REFINE_OFF = 3

_FIT_SIZE_DEFAULT = 7


# =============================================================================
# ピーク精緻化関数 (単体テスト可能・モジュールレベル)
# =============================================================================
def _peak_find_coarse(data_2d, px, py, search_radius):
    """クリック周辺の粗いピーク位置 (2D最大値) を返す。Returns (ix_max, iy_max) integer."""
    ny, nx = data_2d.shape
    ipx, ipy = int(round(px)), int(round(py))
    x0 = max(0, ipx - search_radius)
    x1 = min(nx, ipx + search_radius + 1)
    y0 = max(0, ipy - search_radius)
    y1 = min(ny, ipy + search_radius + 1)
    region = data_2d[y0:y1, x0:x1]
    if region.size == 0:
        return ipx, ipy
    local_y, local_x = np.unravel_index(np.argmax(region), region.shape)
    return x0 + local_x, y0 + local_y


def peak_refine_parabola_3d(volume, iz, iy, ix):
    """各軸独立パラボラフィット。Returns (x_sub, y_sub, z_sub, log_str)."""
    nz, ny, nx = volume.shape

    def _para(arr, i_c):
        if i_c <= 0 or i_c >= len(arr) - 1:
            return float(i_c)
        im1, i0, ip1 = float(arr[i_c - 1]), float(arr[i_c]), float(arr[i_c + 1])
        denom = im1 - 2.0 * i0 + ip1
        if abs(denom) < 1e-12:
            return float(i_c)
        return i_c + (im1 - ip1) / (2.0 * denom)

    half = 3
    x0, x1 = max(0, ix - half), min(nx, ix + half + 1)
    y0, y1 = max(0, iy - half), min(ny, iy + half + 1)
    z0, z1 = max(0, iz - half), min(nz, iz + half + 1)

    x_sub = x0 + _para(volume[iz, iy, x0:x1], ix - x0)
    y_sub = y0 + _para(volume[iz, y0:y1, ix], iy - y0)
    z_sub = z0 + _para(volume[z0:z1, iy, ix], iz - z0)

    return x_sub, y_sub, z_sub, f"parabola→({x_sub:.3f},{y_sub:.3f},{z_sub:.3f})"


def peak_refine_centroid_3d(volume, iz, iy, ix, fit_size=7):
    """3D重心法 (バックグラウンド除去あり)。Returns (x_sub, y_sub, z_sub, log_str)."""
    nz, ny, nx = volume.shape
    half = fit_size // 2
    z0, z1 = max(0, iz - half), min(nz, iz + half + 1)
    y0, y1 = max(0, iy - half), min(ny, iy + half + 1)
    x0, x1 = max(0, ix - half), min(nx, ix + half + 1)

    region = volume[z0:z1, y0:y1, x0:x1].astype(np.float64)
    region = region - region.min()
    total = region.sum()
    if total < 1e-12:
        return float(ix), float(iy), float(iz), "centroid:zero→fallback"

    zg, yg, xg = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float64)
    x_sub = float((xg * region).sum() / total)
    y_sub = float((yg * region).sum() / total)
    z_sub = float((zg * region).sum() / total)
    return x_sub, y_sub, z_sub, f"centroid→({x_sub:.3f},{y_sub:.3f},{z_sub:.3f})"


def peak_refine_gaussian_3d(volume, iz, iy, ix, fit_size=7):
    """3Dガウシアンフィット。収束失敗またはフィット領域不足でNoneを返す。
    Returns (x_sub, y_sub, z_sub, sx, sy, sz, log_str) or None."""
    from scipy.optimize import curve_fit

    nz, ny, nx = volume.shape
    half = fit_size // 2
    z0, z1 = max(0, iz - half), min(nz, iz + half + 1)
    y0, y1 = max(0, iy - half), min(ny, iy + half + 1)
    x0, x1 = max(0, ix - half), min(nx, ix + half + 1)

    region = volume[z0:z1, y0:y1, x0:x1].astype(np.float64)
    if region.size < 8:
        return None
    region_min, region_max = float(region.min()), float(region.max())
    if region_max - region_min < 1e-12:
        return None
    if region.size < (fit_size ** 3) * 0.4:
        return None

    zg, yg, xg = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float64)
    xf, yf, zf = xg.ravel(), yg.ravel(), zg.ravel()
    data = region.ravel()

    def _gauss3d(coords, A, cx, cy, cz, sx, sy, sz, B):
        xc, yc, zc = coords
        return A * np.exp(
            -((xc - cx) ** 2 / (2 * sx ** 2)
              + (yc - cy) ** 2 / (2 * sy ** 2)
              + (zc - cz) ** 2 / (2 * sz ** 2))
        ) + B

    A0 = region_max - region_min
    p0 = [A0, float(ix), float(iy), float(iz), 1.5, 1.5, 1.5, region_min]
    lo = [0.0, float(x0) - 0.5, float(y0) - 0.5, float(z0) - 0.5,
          0.2, 0.2, 0.2, -np.inf]
    hi = [np.inf, float(x1) + 0.5, float(y1) + 0.5, float(z1) + 0.5,
          float(fit_size) * 2, float(fit_size) * 2, float(fit_size) * 2, np.inf]

    try:
        popt, _ = curve_fit(_gauss3d, (xf, yf, zf), data,
                            p0=p0, bounds=(lo, hi), maxfev=2000)
        x_fit, y_fit, z_fit = popt[1], popt[2], popt[3]
        sx_fit, sy_fit, sz_fit = popt[4], popt[5], popt[6]

        if not (x0 - 0.5 <= x_fit <= x1 + 0.5
                and y0 - 0.5 <= y_fit <= y1 + 0.5
                and z0 - 0.5 <= z_fit <= z1 + 0.5):
            return None

        log = (f"gaussian→({x_fit:.3f},{y_fit:.3f},{z_fit:.3f}) "
               f"σ=({sx_fit:.2f},{sy_fit:.2f},{sz_fit:.2f})")
        return x_fit, y_fit, z_fit, sx_fit, sy_fit, sz_fit, log
    except Exception:
        return None


def _apply_peak_refinement_on_region(volume, iz, iy, ix, refine_mode, fit_size):
    """指定アルゴリズムで体積領域内のピーク精緻化を行い (x_sub, y_sub, z_sub) を返す。"""
    if refine_mode == PEAK_REFINE_GAUSSIAN:
        result = peak_refine_gaussian_3d(volume, iz, iy, ix, fit_size)
        if result is None:
            r = peak_refine_parabola_3d(volume, iz, iy, ix)
            return r[0], r[1], r[2]
        return result[0], result[1], result[2]
    if refine_mode == PEAK_REFINE_CENTROID:
        r = peak_refine_centroid_3d(volume, iz, iy, ix, fit_size)
        return r[0], r[1], r[2]
    if refine_mode == PEAK_REFINE_PARABOLA:
        r = peak_refine_parabola_3d(volume, iz, iy, ix)
        return r[0], r[1], r[2]
    return float(ix), float(iy), float(iz)


# =============================================================================
# 3D-AIR-IMAGE ウィンドウ選択ダイアログ
# =============================================================================
class AirWindowSelectDialog(QDialog):
    """3D-AIR-IMAGEのウィンドウ一覧から選択するダイアログ"""

    def __init__(self, parent, api, mode="volume"):
        """
        mode: "volume" = 全ウィンドウ表示（VolumeImage選択用）
              "series" = Seriesウィンドウのみ
        """
        super().__init__(parent)
        self.api = api
        self.mode = mode
        self.selected_wid = None
        self.selected_caption = ""
        self.setWindowTitle("3D-AIR-IMAGE ウィンドウ選択")
        self.setMinimumSize(550, 400)
        self.setStyleSheet("""
            QDialog { background-color: #f5f7fa; }
            QLabel { color: #4a6880; font-size: 12px;
                     font-family: 'Segoe UI', 'Yu Gothic UI', sans-serif; }
            QListWidget {
                background-color: #ffffff; color: #1a2a3a;
                border: 1px solid #c8daea; border-left: 2px solid #0077b6;
                border-radius: 6px; font-family: Consolas; font-size: 12px;
                outline: none;
            }
            QListWidget::item { padding: 6px 12px; }
            QListWidget::item:hover { background-color: #eaf2fb; color: #003a70; }
            QListWidget::item:selected {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #cce8ff, stop:1 #d8ccff);
                color: #003a70;
            }
            QPushButton {
                background-color: #eaf2fb; color: #2a6090;
                border: 1px solid #b8d4ea; border-radius: 6px;
                padding: 7px 16px; font-weight: bold; font-size: 12px;
                font-family: 'Segoe UI', 'Yu Gothic UI', sans-serif;
            }
            QPushButton:hover { background-color: #d4eaf8; border-color: #0077b6; color: #004a80; }
        """)

        layout = QVBoxLayout(self)

        if mode == "series":
            desc = "Seriesウィンドウを選択してください（2Dスライスを積み上げて3D化）:"
        else:
            desc = "ウィンドウを選択してください:"
        layout.addWidget(QLabel(desc))

        self.btn_refresh = QPushButton("🔄 リスト更新")
        self.btn_refresh.clicked.connect(self._refresh_list)
        layout.addWidget(self.btn_refresh)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.list_widget)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #a6adc8; font-size: 13px;")
        layout.addWidget(self.lbl_info)

        btn_row = QHBoxLayout()
        self.btn_ok = QPushButton("選択")
        self.btn_ok.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0077b6, stop:1 #7209b7);
            color: #ffffff; font-weight: bold; border: none;
            border-radius: 6px; padding: 7px 16px; font-size: 12px;
        """)
        self.btn_ok.clicked.connect(self._on_accept)
        btn_cancel = QPushButton("キャンセル")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

        self.windows = []
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        self.windows = []
        try:
            all_windows = self.api.window_get_list_wid_and_caption()

            if self.mode == "series":
                for w in all_windows:
                    try:
                        if self.api.window_is_mode_series_get(w["wid"]):
                            self.windows.append(w)
                    except Exception:
                        pass
            else:
                self.windows = all_windows

            for w in self.windows:
                item = QListWidgetItem(f"ID:{w['wid']}  |  {w['caption']}")
                self.list_widget.addItem(item)

            self.lbl_info.setText(f"{len(self.windows)} 個のウィンドウが見つかりました")
        except Exception as e:
            self.lbl_info.setText(f"エラー: {e}")

    def _on_double_click(self, item):
        self._on_accept()

    def _on_accept(self):
        idx = self.list_widget.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "警告", "ウィンドウを選択してください")
            return
        self.selected_wid = self.windows[idx]["wid"]
        self.selected_caption = self.windows[idx]["caption"]
        self.accept()


# =============================================================================
# スライスキャンバス
# =============================================================================
class SliceCanvas(FigureCanvas):
    point_clicked = pyqtSignal(float, float)

    def __init__(self, parent=None, title="Slice"):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor('#f5f7fa')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08)
        super().__init__(self.fig)
        self.setParent(parent)

        self.title = title
        self.ax.set_title(title, color='#0077b6', fontsize=13, fontweight='bold',
                          fontfamily='Segoe UI')
        self.ax.tick_params(colors='#4a6880', labelsize=10)
        for spine in self.ax.spines.values():
            spine.set_color('#d0dde8')
            spine.set_linewidth(0.8)

        self.img_handle = None
        self.atom_circles = []
        self.click_markers = []
        self.distance_lines = []
        self.distance_texts = []
        self.angle_lines = []
        self.angle_texts = []
        self.peak_markers = []

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:
            self.point_clicked.emit(event.xdata, event.ydata)

    def display_slice(self, data_2d, extent, cmap='hot', vmin=None, vmax=None):
        if self.img_handle is not None:
            self.img_handle.remove()
        self.img_handle = self.ax.imshow(
            data_2d, extent=extent, origin='lower',
            cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax,
            interpolation='bilinear'
        )
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.draw_idle()

    def overlay_atoms(self, positions, colors, radii):
        """positions: [(x,y),...], colors: ['#hex',...], radii: [float,...]"""
        self.clear_atoms()
        for (px, py), color, r in zip(positions, colors, radii):
            circle = Circle(
                (px, py), r, fill=False, edgecolor=color,
                linewidth=1.5, linestyle='-', alpha=0.9
            )
            self.ax.add_patch(circle)
            self.atom_circles.append(circle)
        self.draw_idle()

    def clear_atoms(self):
        for obj in self.atom_circles:
            obj.remove()
        self.atom_circles.clear()
        self.draw_idle()

    def add_click_marker(self, x, y, label="", color='#0077b6'):
        marker = self.ax.plot(x, y, '+', color=color, markersize=15,
                              markeredgewidth=2, zorder=10)[0]
        self.click_markers.append(marker)
        if label:
            txt = self.ax.text(x + 0.1, y + 0.1, label, color=color,
                               fontsize=9, fontweight='bold', zorder=10)
            self.click_markers.append(txt)
        self.draw_idle()

    def add_distance_line(self, x1, y1, x2, y2, dist_text, color='#00875a'):
        line = Line2D([x1, x2], [y1, y2], color=color, linewidth=1.5,
                      linestyle='--', zorder=9)
        self.ax.add_line(line)
        self.distance_lines.append(line)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        txt = self.ax.text(
            mx, my + 0.1, dist_text, color=color, fontsize=10,
            fontweight='bold', ha='center', va='bottom', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff',
                      edgecolor=color, alpha=0.95)
        )
        self.distance_texts.append(txt)
        self.draw_idle()

    def clear_canvas_markers(self):
        """データは消さずキャンバス上のマーカー描画のみ削除"""
        for lst in (self.click_markers, self.distance_lines, self.distance_texts,
                    self.angle_lines, self.angle_texts):
            for obj in lst:
                try:
                    obj.remove()
                except Exception:
                    pass
            lst.clear()

    def clear_measurements(self):
        self.clear_canvas_markers()
        self.draw_idle()

    def add_angle_lines(self, x1, y1, x2, y2, x3, y3, angle_deg):
        """Draw P1-P2, P2-P3 segments with labeled markers and angle text at vertex P2."""
        line_color = '#e08d3c'
        for xs, ys in (([x1, x2], [y1, y2]), ([x2, x3], [y2, y3])):
            line = Line2D(xs, ys, color=line_color, linewidth=1.5, linestyle='--', zorder=9)
            self.ax.add_line(line)
            self.angle_lines.append(line)
        for x, y, lbl, col in [(x1, y1, 'A1', '#89b4fa'),
                                 (x2, y2, 'A2', '#ff79c6'),
                                 (x3, y3, 'A3', '#f9e2af')]:
            mk = self.ax.plot(x, y, '+', color=col, markersize=15,
                              markeredgewidth=2, zorder=10)[0]
            self.click_markers.append(mk)
            txt = self.ax.text(x + 0.1, y + 0.1, lbl, color=col,
                               fontsize=9, fontweight='bold', zorder=10)
            self.click_markers.append(txt)
        atxt = self.ax.text(
            x2, y2 - 0.15, f"{angle_deg:.1f}°",
            color=line_color, fontsize=10, fontweight='bold',
            ha='center', va='top', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff',
                      edgecolor=line_color, alpha=0.95)
        )
        self.angle_texts.append(atxt)
        self.draw_idle()

    def draw_peak_markers(self, peaks_xy):
        """Draw detected peaks as cyan scatter markers at (x, y) positions in Å."""
        self.clear_peak_markers()
        if not peaks_xy:
            return
        xs = [p[0] for p in peaks_xy]
        ys = [p[1] for p in peaks_xy]
        sc = self.ax.scatter(xs, ys, s=35, c='#00ffcc', marker='o',
                             alpha=0.55, linewidths=0.8, edgecolors='#009966',
                             zorder=8)
        self.peak_markers.append(sc)
        self.draw_idle()

    def clear_peak_markers(self):
        """Remove only auto-detected peak markers (keep measurement markers)."""
        for obj in self.peak_markers:
            try:
                obj.remove()
            except Exception:
                pass
        self.peak_markers.clear()


# =============================================================================
# メインウィンドウ
# =============================================================================
# =============================================================================
# 非同期ピーク検出ワーカー
# =============================================================================
class PeakDetectionWorker(QThread):
    finished = pyqtSignal(list)

    def __init__(self, volume_region, z_offset, metadata, percentile, min_dist_ang,
                 refine_mode=PEAK_REFINE_OFF, fit_size=_FIT_SIZE_DEFAULT):
        super().__init__()
        self.volume_region = volume_region
        self.z_offset = z_offset
        self.metadata = metadata
        self.percentile = percentile
        self.min_dist_ang = min_dist_ang
        self.refine_mode = refine_mode
        self.fit_size = fit_size

    def run(self):
        m = self.metadata
        dz_v, dy_v, dx_v = m['voxel_size']
        oz, oy, ox = m['z_range'][0], m['y_range'][0], m['x_range'][0]
        mda = self.min_dist_ang
        filter_size = (
            max(1, 2 * int(round(mda / dz_v)) + 1),
            max(1, 2 * int(round(mda / dy_v)) + 1),
            max(1, 2 * int(round(mda / dx_v)) + 1),
        )
        vol = self.volume_region.astype(np.float32)
        filtered = maximum_filter(vol, size=filter_size)
        threshold = np.percentile(vol, self.percentile)
        local_max = (vol == filtered) & (vol > threshold)
        zz, yy, xx = np.where(local_max)
        peaks = []
        for zi, yi, xi in zip(zz, yy, xx):
            abs_zi = int(zi) + self.z_offset
            if self.refine_mode != PEAK_REFINE_OFF:
                x_sub, y_sub, z_sub = _apply_peak_refinement_on_region(
                    vol, int(zi), int(yi), int(xi), self.refine_mode, self.fit_size)
                abs_z_sub = z_sub + self.z_offset
            else:
                x_sub, y_sub, abs_z_sub = float(xi), float(yi), float(abs_zi)
            peaks.append({
                'x': ox + x_sub * dx_v,
                'y': oy + y_sub * dy_v,
                'z': oz + abs_z_sub * dz_v,
                'px': x_sub, 'py': y_sub, 'pz': abs_z_sub,
                'intensity': float(vol[int(zi), int(yi), int(xi)]),
            })
        self.finished.emit(peaks)


class AtomViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atom Image Viewer - 原子像再構成ビューア")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(self._build_stylesheet())

        # 3D-AIR-IMAGE API
        self.air = connect_air_api()

        # データ
        self.volume = None
        self.metadata = None
        self.atom_sets: list = []      # 読み込んだXYZセット一覧
        self.active_atom_set_index: int = -1
        self.click_points = []
        self.measurements = []
        self.angle_measurements = []
        self.detected_peaks = []
        self.measure_type = 'distance'
        self._peak_worker = None
        self.element_settings = {}   # 後方互換 (アクティブセットへの参照で上書き)

        self._build_ui()
        self._connect_signals()

        if self.air:
            self.statusBar().showMessage(
                "3D-AIR-IMAGE API 接続済み — ボリュームデータを読み込んでください")
        else:
            self.statusBar().showMessage(
                "API未接続 — ファイルから読み込むか、3D-AIR-IMAGEを起動してください")

    def _build_stylesheet(self):
        return """
        QMainWindow { background-color: #f5f7fa; }
        QWidget {
            background-color: #f5f7fa; color: #1a2a3a; font-size: 13px;
            font-family: 'Segoe UI', 'Yu Gothic UI', sans-serif;
        }
        QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }

        /* ── GroupBox ── */
        QGroupBox {
            background-color: #ffffff;
            border: 1px solid #d8e4f0;
            border-left: 2px solid #0077b6;
            border-radius: 8px;
            margin-top: 16px;
            padding: 14px 10px 10px 12px;
            font-weight: bold; font-size: 12px; color: #0077b6;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 12px; padding: 0 6px;
            color: #0077b6; background-color: #ffffff;
            font-size: 11px; letter-spacing: 1px;
        }

        /* ── 通常ボタン ── */
        QPushButton {
            background-color: #eaf2fb;
            color: #2a6090;
            border: 1px solid #b8d4ea;
            border-radius: 6px;
            padding: 7px 14px; font-weight: bold; font-size: 12px;
        }
        QPushButton:hover {
            background-color: #d4eaf8;
            border: 1px solid #0077b6;
            color: #004a80;
        }
        QPushButton:pressed {
            background-color: #b8d8f0;
            border-color: #005a90;
        }
        QPushButton:checked {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #cce8ff, stop:1 #d8ccff);
            border: 1px solid #0077b6; color: #003a70;
        }

        /* ── Primary: ブルー → バイオレット ── */
        QPushButton#primary {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0077b6, stop:1 #7209b7);
            color: #ffffff; border: none; border-radius: 6px;
        }
        QPushButton#primary:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0096d6, stop:1 #9b2de0);
        }
        QPushButton#primary:checked {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #00aaee, stop:1 #b44dff);
            border: none;
        }

        /* ── Danger ── */
        QPushButton#danger {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #e53935, stop:1 #c62828);
            color: #ffffff; border: none; border-radius: 6px;
        }
        QPushButton#danger:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #ff5252, stop:1 #e53935);
        }

        /* ── AIR: ティール ── */
        QPushButton#air {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #00796b, stop:1 #00897b);
            color: #ffffff; border: none; border-radius: 6px;
        }
        QPushButton#air:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #00897b, stop:1 #00bfa5);
        }
        QPushButton#air:disabled {
            background: #e0eee8; color: #90b0a8; border: 1px solid #c0dcd4;
        }

        /* ── スライダー ── */
        QSlider::groove:horizontal {
            height: 3px; background: #d0dde8; border-radius: 2px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0077b6, stop:1 #7209b7);
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #0077b6; border: 2px solid #005a90;
            width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #0096d6; border-color: #0077b6;
        }

        /* ── 入力 ── */
        QSpinBox, QDoubleSpinBox {
            background-color: #ffffff; border: 1px solid #c8daea;
            border-radius: 5px; padding: 4px 6px; color: #1a2a3a; font-size: 12px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #0077b6; }
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            background: #eaf2fb; border: none; width: 16px;
        }

        QComboBox {
            background-color: #ffffff; border: 1px solid #c8daea;
            border-radius: 5px; padding: 4px 8px; color: #1a2a3a; font-size: 12px;
        }
        QComboBox:focus { border-color: #0077b6; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background-color: #ffffff; border: 1px solid #c8daea;
            color: #1a2a3a; selection-background-color: #cce8ff;
            selection-color: #003a70; outline: none; padding: 2px;
        }

        /* ── ラベル ── */
        QLabel { color: #4a6880; font-size: 12px; }
        QLabel#heading { font-size: 18px; font-weight: bold; color: #0077b6; }

        /* ── ステータスバー ── */
        QStatusBar {
            background-color: #e8f0f8; color: #4a6880;
            font-size: 11px; border-top: 1px solid #c8daea;
        }
        QStatusBar::item { border: none; }

        /* ── テーブル ── */
        QTableWidget {
            background-color: #ffffff; gridline-color: #d8e8f4;
            border: 1px solid #c8daea; border-radius: 6px; font-size: 12px;
            alternate-background-color: #f0f7fd;
        }
        QTableWidget::item { padding: 4px; color: #2a4a60; }
        QTableWidget::item:selected { background-color: #cce8ff; color: #003a70; }
        QHeaderView::section {
            background-color: #eaf2fb; color: #0077b6;
            border: none; border-bottom: 1px solid #c8daea;
            padding: 5px; font-weight: bold; font-size: 11px;
            letter-spacing: 1px;
        }

        /* ── チェックボックス ── */
        QCheckBox { spacing: 8px; font-size: 12px; color: #3a6080; }
        QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; }
        QCheckBox::indicator:unchecked {
            border: 1px solid #b8d0e8; background-color: #ffffff;
        }
        QCheckBox::indicator:unchecked:hover { border-color: #0077b6; }
        QCheckBox::indicator:checked {
            border: 1px solid #0077b6;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 #0077b6, stop:1 #7209b7);
        }

        /* ── スクロールバー ── */
        QScrollBar:vertical {
            background: #eaf2fb; width: 6px; border-radius: 3px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #b8d4ea; border-radius: 3px; min-height: 24px;
        }
        QScrollBar::handle:vertical:hover { background: #0077b6; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: #eaf2fb; height: 6px; border-radius: 3px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #b8d4ea; border-radius: 3px; min-width: 24px;
        }
        QScrollBar::handle:horizontal:hover { background: #0077b6; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

        /* ── フレーム区切り線 ── */
        QFrame[frameShape="4"] {
            border: none; border-top: 1px solid #d8e8f4; margin: 4px 0;
        }
        """

    # -------------------------------------------------------------------------
    # UI構築
    # -------------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 0, 4)
        main_layout.setSpacing(4)

        # === 左パネル (スクロール可能) ===
        left_content = QWidget()
        left_content.setMinimumWidth(380)
        left_layout = QVBoxLayout(left_content)
        left_layout.setSpacing(4)

        title_label = QLabel("◈  ATOM IMAGE VIEWER")
        title_label.setObjectName("heading")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 16px; font-weight: bold; padding: 12px 6px 10px 6px;
            color: #0077b6;
            border-bottom: 1px solid #d0e8f8;
            letter-spacing: 3px;
            font-family: 'Segoe UI', 'Yu Gothic UI', sans-serif;
        """)
        left_layout.addWidget(title_label)

        # --- データ読み込み ---
        file_group = QGroupBox("▸ DATA SOURCE")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(4)

        self.btn_load_from_air = QPushButton("3D-AIR-IMAGE から取得")
        self.btn_load_from_air.setObjectName("air")
        self.btn_load_from_air.setEnabled(self.air is not None)
        file_layout.addWidget(self.btn_load_from_air)

        self.btn_load_series = QPushButton("Series → 3Dボリューム構築")
        self.btn_load_series.setObjectName("air")
        self.btn_load_series.setEnabled(self.air is not None)
        file_layout.addWidget(self.btn_load_series)

        self.btn_load_file = QPushButton("ファイルから読み込み (.npy/.npz/.xml_air)")
        self.btn_load_file.setObjectName("primary")
        file_layout.addWidget(self.btn_load_file)

        self.lbl_volume_info = QLabel("— 未読み込み —")
        self.lbl_volume_info.setStyleSheet("color: #90a8c0; font-size: 11px;")
        self.lbl_volume_info.setWordWrap(True)
        file_layout.addWidget(self.lbl_volume_info)

        self.btn_reconnect = QPushButton("↺  API再接続")
        self.btn_reconnect.setStyleSheet("font-size: 11px; padding: 5px 10px; color: #4a7a9a;")
        file_layout.addWidget(self.btn_reconnect)

        self.btn_redraw_volume = QPushButton("⟳  表示を再描画")
        self.btn_redraw_volume.setToolTip(
            "描画キャッシュをクリアし、現スライスを再生成します。\n"
            "ビューポート (拡大率・パン) を初期状態に戻します。\n"
            "データ・測定履歴は維持されます。")
        self.btn_redraw_volume.setStyleSheet("font-size: 11px; padding: 5px 10px;")
        file_layout.addWidget(self.btn_redraw_volume)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("border-top: 1px solid #061828;")
        file_layout.addWidget(sep)

        self.btn_load_xyz = QPushButton("XYZ読み込み (複数可)")
        file_layout.addWidget(self.btn_load_xyz)

        self.list_xyz_files = QListWidget()
        self.list_xyz_files.setMaximumHeight(120)
        self.list_xyz_files.setStyleSheet(
            "QListWidget { border: 1px solid #c8daea; border-radius: 4px; "
            "font-size: 11px; background: #fff; color: #1a2a3a; }"
            "QListWidget::item { padding: 3px 6px; }"
            "QListWidget::item:selected { background: #cce8ff; color: #003a70; }"
        )
        file_layout.addWidget(self.list_xyz_files)

        xyz_btn_row2 = QHBoxLayout()
        self.btn_remove_xyz = QPushButton("削除")
        self.btn_remove_xyz.setObjectName("danger")
        self.btn_remove_xyz.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        self.btn_show_all_xyz = QPushButton("全表示")
        self.btn_show_all_xyz.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        self.btn_hide_all_xyz = QPushButton("全非表示")
        self.btn_hide_all_xyz.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        xyz_btn_row2.addWidget(self.btn_remove_xyz)
        xyz_btn_row2.addWidget(self.btn_show_all_xyz)
        xyz_btn_row2.addWidget(self.btn_hide_all_xyz)
        file_layout.addLayout(xyz_btn_row2)

        self.lbl_xyz_info = QLabel("読み込み済み: 0 / 表示中: 0")
        self.lbl_xyz_info.setStyleSheet("color: #90a8c0; font-size: 11px;")
        self.lbl_xyz_info.setWordWrap(True)
        file_layout.addWidget(self.lbl_xyz_info)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # --- Atomオフセット (アクティブファイル) ---
        offset_group = QGroupBox("▸ ATOM OFFSET (アクティブファイル)")
        offset_layout = QGridLayout()
        self.lbl_active_set_name = QLabel("編集中: —")
        self.lbl_active_set_name.setStyleSheet(
            "color: #0077b6; font-size: 11px; font-style: italic;")
        offset_layout.addWidget(self.lbl_active_set_name, 0, 0, 1, 2)
        offset_layout.addWidget(QLabel("X offset (Å):"), 1, 0)
        self.spin_offset_x = QDoubleSpinBox()
        self.spin_offset_x.setRange(-100.0, 100.0)
        self.spin_offset_x.setDecimals(3)
        self.spin_offset_x.setValue(0.0)
        self.spin_offset_x.setSingleStep(0.05)
        self.spin_offset_x.setEnabled(False)
        offset_layout.addWidget(self.spin_offset_x, 1, 1)
        offset_layout.addWidget(QLabel("Y offset (Å):"), 2, 0)
        self.spin_offset_y = QDoubleSpinBox()
        self.spin_offset_y.setRange(-100.0, 100.0)
        self.spin_offset_y.setDecimals(3)
        self.spin_offset_y.setValue(0.0)
        self.spin_offset_y.setSingleStep(0.05)
        self.spin_offset_y.setEnabled(False)
        offset_layout.addWidget(self.spin_offset_y, 2, 1)
        offset_layout.addWidget(QLabel("Z offset (Å):"), 3, 0)
        self.spin_offset_z = QDoubleSpinBox()
        self.spin_offset_z.setRange(-100.0, 100.0)
        self.spin_offset_z.setDecimals(3)
        self.spin_offset_z.setValue(0.0)
        self.spin_offset_z.setSingleStep(0.05)
        self.spin_offset_z.setEnabled(False)
        offset_layout.addWidget(self.spin_offset_z, 3, 1)
        self.btn_reset_offset = QPushButton("リセット (0, 0, 0)")
        self.btn_reset_offset.setEnabled(False)
        offset_layout.addWidget(self.btn_reset_offset, 4, 0, 1, 2)
        offset_group.setLayout(offset_layout)
        left_layout.addWidget(offset_group)

        # --- ボクセルサイズ ---
        voxel_group = QGroupBox("▸ VOXEL SIZE (Å)")
        voxel_layout = QGridLayout()
        self.spin_dx = self._make_voxel_spin(voxel_layout, "dx:", 0)
        self.spin_dy = self._make_voxel_spin(voxel_layout, "dy:", 1)
        self.spin_dz = self._make_voxel_spin(voxel_layout, "dz:", 2)
        self.btn_apply_voxel = QPushButton("適用")
        voxel_layout.addWidget(self.btn_apply_voxel, 3, 0, 1, 2)
        voxel_group.setLayout(voxel_layout)
        left_layout.addWidget(voxel_group)

        # --- スライス制御 ---
        slice_group = QGroupBox("▸ SLICE (ホログラム表示位置)")
        slice_layout = QGridLayout()

        slice_layout.addWidget(QLabel("Z (Å):"), 0, 0)
        self.slider_z = QSlider(Qt.Horizontal)
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setDecimals(3)
        self.spin_z.setSingleStep(0.1)
        self.spin_z.setRange(-9999.0, 9999.0)
        slice_layout.addWidget(self.slider_z, 0, 1)
        slice_layout.addWidget(self.spin_z, 0, 2)

        slice_layout.addWidget(QLabel("Y (Å):"), 1, 0)
        self.slider_y = QSlider(Qt.Horizontal)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setDecimals(3)
        self.spin_y.setSingleStep(0.1)
        self.spin_y.setRange(-9999.0, 9999.0)
        slice_layout.addWidget(self.slider_y, 1, 1)
        slice_layout.addWidget(self.spin_y, 1, 2)

        slice_layout.addWidget(QLabel("X (Å):"), 2, 0)
        self.slider_x = QSlider(Qt.Horizontal)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setDecimals(3)
        self.spin_x.setSingleStep(0.1)
        self.spin_x.setRange(-9999.0, 9999.0)
        slice_layout.addWidget(self.slider_x, 2, 1)
        slice_layout.addWidget(self.spin_x, 2, 2)

        slice_group.setLayout(slice_layout)
        left_layout.addWidget(slice_group)

        # --- クラスタースライス ---
        cluster_group = QGroupBox("▸ CLUSTER SLICE (原子表示基準位置)")
        cluster_layout = QGridLayout()

        self.chk_cluster_link_hologram = QCheckBox("ホログラムに連動")
        self.chk_cluster_link_hologram.setChecked(True)
        cluster_layout.addWidget(self.chk_cluster_link_hologram, 0, 0, 1, 3)

        self.spin_cluster_pos = {}
        self.slider_cluster = {}
        for _row_i, (_axis, _label) in enumerate(
                [('z', 'Z (Å):'), ('y', 'Y (Å):'), ('x', 'X (Å):')], start=1):
            cluster_layout.addWidget(QLabel(_label), _row_i, 0)
            _sl = QSlider(Qt.Horizontal)
            _sl.setEnabled(False)
            _sp = QDoubleSpinBox()
            _sp.setDecimals(3)
            _sp.setSingleStep(0.1)
            _sp.setRange(-9999.0, 9999.0)
            _sp.setEnabled(False)
            cluster_layout.addWidget(_sl, _row_i, 1)
            cluster_layout.addWidget(_sp, _row_i, 2)
            self.slider_cluster[_axis] = _sl
            self.spin_cluster_pos[_axis] = _sp

        self.btn_cluster_sync = QPushButton("現在のホログラム位置に合わせる")
        cluster_layout.addWidget(self.btn_cluster_sync, 4, 0, 1, 3)

        cluster_group.setLayout(cluster_layout)
        left_layout.addWidget(cluster_group)

        # --- 表示設定 ---
        display_group = QGroupBox("▸ RENDER")
        display_layout = QVBoxLayout()

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("カラーマップ:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems([
            'hot', 'viridis', 'plasma', 'inferno', 'magma',
            'cividis', 'gray', 'bone', 'copper', 'YlOrRd',
        ])
        cmap_row.addWidget(self.combo_cmap)
        display_layout.addLayout(cmap_row)

        self.chk_show_atoms = QCheckBox("原子位置を表示")
        self.chk_show_atoms.setChecked(True)
        display_layout.addWidget(self.chk_show_atoms)

        display_group.setLayout(display_layout)
        left_layout.addWidget(display_group)

        # --- 強度調整 ---
        intensity_group = QGroupBox("▸ INTENSITY")
        intensity_layout = QVBoxLayout()
        intensity_layout.setSpacing(4)

        # 自動 / 手動 切り替え
        self.chk_auto_intensity = QCheckBox("パーセンタイル自動スケール")
        self.chk_auto_intensity.setChecked(True)
        intensity_layout.addWidget(self.chk_auto_intensity)

        # パーセンタイル行
        pct_row = QHBoxLayout()
        pct_row.addWidget(QLabel("下限:"))
        self.spin_pct_low = QDoubleSpinBox()
        self.spin_pct_low.setRange(0.0, 98.9)
        self.spin_pct_low.setDecimals(1)
        self.spin_pct_low.setValue(0.1)
        self.spin_pct_low.setSingleStep(0.1)
        self.spin_pct_low.setFixedWidth(60)
        pct_row.addWidget(self.spin_pct_low)
        pct_row.addWidget(QLabel("%  上限:"))
        self.spin_pct_high = QDoubleSpinBox()
        self.spin_pct_high.setRange(1.0, 100.0)
        self.spin_pct_high.setDecimals(1)
        self.spin_pct_high.setValue(99.9)
        self.spin_pct_high.setSingleStep(0.1)
        self.spin_pct_high.setFixedWidth(60)
        pct_row.addWidget(self.spin_pct_high)
        pct_row.addWidget(QLabel("%"))
        pct_row.addStretch()
        intensity_layout.addLayout(pct_row)

        # 手動 vmin / vmax
        vmin_row = QHBoxLayout()
        vmin_row.addWidget(QLabel("vmin:"))
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e9, 1e9)
        self.spin_vmin.setDecimals(5)
        self.spin_vmin.setValue(0.0)
        self.spin_vmin.setEnabled(False)
        vmin_row.addWidget(self.spin_vmin)
        intensity_layout.addLayout(vmin_row)

        vmax_row = QHBoxLayout()
        vmax_row.addWidget(QLabel("vmax:"))
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e9, 1e9)
        self.spin_vmax.setDecimals(5)
        self.spin_vmax.setValue(1.0)
        self.spin_vmax.setEnabled(False)
        vmax_row.addWidget(self.spin_vmax)
        intensity_layout.addLayout(vmax_row)

        # 設定ボタン行
        ibtn_row = QHBoxLayout()
        self.btn_intensity_from_slice = QPushButton("現スライス")
        self.btn_intensity_from_slice.setStyleSheet("font-size: 11px; padding: 5px 8px;")
        self.btn_intensity_from_volume = QPushButton("全体")
        self.btn_intensity_from_volume.setStyleSheet("font-size: 11px; padding: 5px 8px;")
        ibtn_row.addWidget(self.btn_intensity_from_slice)
        ibtn_row.addWidget(self.btn_intensity_from_volume)
        intensity_layout.addLayout(ibtn_row)

        intensity_group.setLayout(intensity_layout)
        left_layout.addWidget(intensity_group)

        # --- 元素設定 ---
        elem_group = QGroupBox("▸ ELEMENTS")
        elem_outer_layout = QVBoxLayout()

        self.lbl_active_elem_set = QLabel("編集中: —")
        self.lbl_active_elem_set.setStyleSheet(
            "color: #0077b6; font-size: 11px; font-style: italic;")
        elem_outer_layout.addWidget(self.lbl_active_elem_set)

        # ヘッダー行
        hdr = QHBoxLayout()
        for txt, w in [("ELEM", 42), ("COLOR", 52), ("R (Å)", 72)]:
            lb = QLabel(txt)
            lb.setFixedWidth(w)
            lb.setStyleSheet("color: #0077b6; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
            hdr.addWidget(lb)
        hdr.addStretch()
        elem_outer_layout.addLayout(hdr)

        # 元素行を入れるスクロール可能エリア
        self.elem_rows_widget = QWidget()
        self.elem_rows_layout = QVBoxLayout(self.elem_rows_widget)
        self.elem_rows_layout.setSpacing(3)
        self.elem_rows_layout.setContentsMargins(0, 0, 0, 0)

        elem_scroll = QScrollArea()
        elem_scroll.setWidget(self.elem_rows_widget)
        elem_scroll.setWidgetResizable(True)
        elem_scroll.setMaximumHeight(200)
        elem_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        elem_scroll.setStyleSheet("QScrollArea { border: none; }")
        elem_outer_layout.addWidget(elem_scroll)

        elem_group.setLayout(elem_outer_layout)
        left_layout.addWidget(elem_group)

        # --- 距離・角度測定 ---
        measure_group = QGroupBox("▸ MEASURE")
        measure_layout = QVBoxLayout()

        self.lbl_measure_mode = QLabel("● MEASURE  OFF")
        self.lbl_measure_mode.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #90a8c0; letter-spacing: 2px;"
        )
        measure_layout.addWidget(self.lbl_measure_mode)

        self.btn_toggle_measure = QPushButton("◉  MEASURE MODE")
        self.btn_toggle_measure.setObjectName("primary")
        self.btn_toggle_measure.setCheckable(True)
        measure_layout.addWidget(self.btn_toggle_measure)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("測定タイプ:"))
        self.radio_dist = QRadioButton("2点距離")
        self.radio_angle = QRadioButton("3点角度")
        self.radio_dist.setChecked(True)
        self.btn_grp_measure_type = QButtonGroup(self)
        self.btn_grp_measure_type.addButton(self.radio_dist, 0)
        self.btn_grp_measure_type.addButton(self.radio_angle, 1)
        type_row.addWidget(self.radio_dist)
        type_row.addWidget(self.radio_angle)
        type_row.addStretch()
        measure_layout.addLayout(type_row)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("ピーク探索半径 (px):"))
        self.spin_search_radius = QSpinBox()
        self.spin_search_radius.setRange(1, 50)
        self.spin_search_radius.setValue(15)
        search_row.addWidget(self.spin_search_radius)
        measure_layout.addLayout(search_row)

        # --- PEAK REFINEMENT (サブボクセル精度) ---
        refine_sep = QFrame()
        refine_sep.setFrameShape(QFrame.HLine)
        refine_sep.setStyleSheet("border-top: 1px solid #d0dde8; margin: 2px 0;")
        measure_layout.addWidget(refine_sep)

        refine_lbl = QLabel("PEAK REFINEMENT")
        refine_lbl.setStyleSheet(
            "color: #0077b6; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        measure_layout.addWidget(refine_lbl)

        refine_row = QHBoxLayout()
        refine_row.addWidget(QLabel("サブボクセル精度:"))
        self.combo_peak_refine = QComboBox()
        self.combo_peak_refine.addItems([
            "ガウシアンフィット",
            "重心",
            "パラボラ",
            "OFF (従来動作)",
        ])
        self.combo_peak_refine.setCurrentIndex(0)
        self.combo_peak_refine.setToolTip(
            "ガウシアンフィット: 最高精度 (~0.01 Å), scipy.optimize.curve_fit 使用\n"
            "重心: 高速・中精度\n"
            "パラボラ: 軽量フォールバック\n"
            "OFF: 従来の3×3重心のみ (後方互換)")
        refine_row.addWidget(self.combo_peak_refine)
        measure_layout.addLayout(refine_row)

        fit_size_row = QHBoxLayout()
        fit_size_row.addWidget(QLabel("フィット領域 (voxel):"))
        self.spin_fit_size = QSpinBox()
        self.spin_fit_size.setRange(3, 15)
        self.spin_fit_size.setValue(_FIT_SIZE_DEFAULT)
        self.spin_fit_size.setSingleStep(2)
        self.spin_fit_size.setToolTip("奇数のみ有効 (3/5/7/9/11/13/15)\n大きすぎると隣ピークを巻き込む")
        fit_size_row.addWidget(self.spin_fit_size)
        measure_layout.addLayout(fit_size_row)

        self.chk_fit_log = QCheckBox("フィット詳細ログ (デバッグ)")
        self.chk_fit_log.setChecked(False)
        self.chk_fit_log.setToolTip("ONにするとクリック時に精緻化詳細をステータスバーに表示")
        measure_layout.addWidget(self.chk_fit_log)

        self.chk_freeze_markers = QCheckBox("マーカー固定 (Zスライスに追従しない)")
        self.chk_freeze_markers.setChecked(False)
        self.chk_freeze_markers.setToolTip(
            "ONにすると測定点マーカーが元のÅ座標に固定されます。\n"
            "Zスライスを変えても位置・角度が変わらないため、\n"
            "別スライスの原子像と重ねて比較できます。")
        measure_layout.addWidget(self.chk_freeze_markers)

        self.btn_clear_measure = QPushButton("✕  CLEAR (現タブ)")
        self.btn_clear_measure.setObjectName("danger")
        measure_layout.addWidget(self.btn_clear_measure)

        measure_group.setLayout(measure_layout)
        left_layout.addWidget(measure_group)

        # --- 測定結果テーブル (3タブ) ---
        result_group = QGroupBox("▸ RESULTS")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(4, 4, 4, 4)

        self.tab_results = QTabWidget()

        # 距離タブ
        dist_tab = QWidget()
        dist_tab_layout = QVBoxLayout(dist_tab)
        dist_tab_layout.setContentsMargins(2, 4, 2, 2)
        self.table_dist = QTableWidget(0, 3)
        self.table_dist.setHorizontalHeaderLabels(["点1 (Å)", "点2 (Å)", "距離 (Å)"])
        self.table_dist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_dist.setMaximumHeight(130)
        dist_tab_layout.addWidget(self.table_dist)
        dist_btn_row = QHBoxLayout()
        self.btn_reset_dist = QPushButton("✕  選択リセット")
        self.btn_reset_dist.setObjectName("danger")
        self.btn_reset_dist.setToolTip("進行中選択 + 距離履歴 + マーカーをすべてクリア")
        dist_btn_row.addWidget(self.btn_reset_dist)
        self.btn_export_dist_csv = QPushButton("↓  CSV EXPORT (距離)")
        dist_btn_row.addWidget(self.btn_export_dist_csv)
        dist_tab_layout.addLayout(dist_btn_row)
        self.tab_results.addTab(dist_tab, "距離")

        # 角度タブ
        angle_tab = QWidget()
        angle_tab_layout = QVBoxLayout(angle_tab)
        angle_tab_layout.setContentsMargins(2, 4, 2, 2)
        self.table_angle = QTableWidget(0, 6)
        self.table_angle.setHorizontalHeaderLabels(
            ["点1 (Å)", "点2/頂点 (Å)", "点3 (Å)", "角度 (°)", "d12 (Å)", "d23 (Å)"])
        self.table_angle.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_angle.setMaximumHeight(130)
        angle_tab_layout.addWidget(self.table_angle)
        angle_btn_row = QHBoxLayout()
        self.btn_reset_angle = QPushButton("✕  選択リセット")
        self.btn_reset_angle.setObjectName("danger")
        self.btn_reset_angle.setToolTip("進行中選択 + 角度履歴 + マーカーをすべてクリア")
        angle_btn_row.addWidget(self.btn_reset_angle)
        self.btn_export_angle_csv = QPushButton("↓  CSV EXPORT (角度)")
        angle_btn_row.addWidget(self.btn_export_angle_csv)
        angle_tab_layout.addLayout(angle_btn_row)
        self.tab_results.addTab(angle_tab, "角度")

        # ピークタブ
        peak_tab = QWidget()
        peak_tab_layout = QVBoxLayout(peak_tab)
        peak_tab_layout.setContentsMargins(2, 4, 2, 2)
        self.table_peaks = QTableWidget(0, 5)
        self.table_peaks.setHorizontalHeaderLabels(
            ["No.", "x (Å)", "y (Å)", "z (Å)", "強度"])
        self.table_peaks.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_peaks.setMaximumHeight(130)
        peak_tab_layout.addWidget(self.table_peaks)
        self.btn_export_peaks_csv = QPushButton("↓  CSV EXPORT (ピーク)")
        peak_tab_layout.addWidget(self.btn_export_peaks_csv)
        self.tab_results.addTab(peak_tab, "ピーク")

        result_layout.addWidget(self.tab_results)
        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)

        # --- ピーク自動検出 ---
        peak_detect_group = QGroupBox("▸ PEAK DETECTION")
        peak_detect_layout = QVBoxLayout()

        peak_detect_layout.addWidget(QLabel("検出範囲:"))
        range_row = QHBoxLayout()
        self.radio_peak_current = QRadioButton("現スライス")
        self.radio_peak_zrange = QRadioButton("Z範囲")
        self.radio_peak_full = QRadioButton("全ボリューム")
        self.radio_peak_current.setChecked(True)
        self.btn_grp_peak_range = QButtonGroup(self)
        self.btn_grp_peak_range.addButton(self.radio_peak_current, 0)
        self.btn_grp_peak_range.addButton(self.radio_peak_zrange, 1)
        self.btn_grp_peak_range.addButton(self.radio_peak_full, 2)
        range_row.addWidget(self.radio_peak_current)
        range_row.addWidget(self.radio_peak_zrange)
        range_row.addWidget(self.radio_peak_full)
        range_row.addStretch()
        peak_detect_layout.addLayout(range_row)

        zrange_row = QHBoxLayout()
        zrange_row.addWidget(QLabel("Z開始 (Å):"))
        self.spin_peak_z_start = QDoubleSpinBox()
        self.spin_peak_z_start.setRange(-1000.0, 1000.0)
        self.spin_peak_z_start.setDecimals(3)
        self.spin_peak_z_start.setValue(0.0)
        self.spin_peak_z_start.setEnabled(False)
        zrange_row.addWidget(self.spin_peak_z_start)
        zrange_row.addWidget(QLabel("終了 (Å):"))
        self.spin_peak_z_end = QDoubleSpinBox()
        self.spin_peak_z_end.setRange(-1000.0, 1000.0)
        self.spin_peak_z_end.setDecimals(3)
        self.spin_peak_z_end.setValue(1.0)
        self.spin_peak_z_end.setEnabled(False)
        zrange_row.addWidget(self.spin_peak_z_end)
        peak_detect_layout.addLayout(zrange_row)

        pct_param_row = QHBoxLayout()
        pct_param_row.addWidget(QLabel("最小強度 (%tile):"))
        self.spin_peak_percentile = QDoubleSpinBox()
        self.spin_peak_percentile.setRange(50.0, 99.99)
        self.spin_peak_percentile.setDecimals(1)
        self.spin_peak_percentile.setValue(99.0)
        self.spin_peak_percentile.setSingleStep(0.5)
        pct_param_row.addWidget(self.spin_peak_percentile)
        peak_detect_layout.addLayout(pct_param_row)

        dist_param_row = QHBoxLayout()
        dist_param_row.addWidget(QLabel("最小ピーク間距離 (Å):"))
        self.spin_peak_min_dist = QDoubleSpinBox()
        self.spin_peak_min_dist.setRange(0.1, 50.0)
        self.spin_peak_min_dist.setDecimals(2)
        self.spin_peak_min_dist.setValue(1.5)
        self.spin_peak_min_dist.setSingleStep(0.1)
        dist_param_row.addWidget(self.spin_peak_min_dist)
        peak_detect_layout.addLayout(dist_param_row)

        self.btn_detect_peaks = QPushButton("▶  ピーク自動検出")
        self.btn_detect_peaks.setObjectName("primary")
        peak_detect_layout.addWidget(self.btn_detect_peaks)

        chk_row = QHBoxLayout()
        self.chk_show_peak_markers = QCheckBox("検出マーカー表示")
        self.chk_show_peak_markers.setChecked(True)
        self.chk_snap_to_peaks = QCheckBox("検出マーカーにスナップ")
        self.chk_snap_to_peaks.setChecked(True)
        chk_row.addWidget(self.chk_show_peak_markers)
        chk_row.addWidget(self.chk_snap_to_peaks)
        peak_detect_layout.addLayout(chk_row)

        peak_detect_group.setLayout(peak_detect_layout)
        left_layout.addWidget(peak_detect_group)

        left_layout.addStretch()

        # 左パネルをスクロール可能にラップ
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_content)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(420)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # === 右パネル ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.canvas_xy = SliceCanvas(title="XY平面 (Z固定)")
        self.canvas_xz = SliceCanvas(title="XZ平面 (Y固定)")
        self.canvas_yz = SliceCanvas(title="YZ平面 (X固定)")

        self.slice_tab = QTabWidget()
        self.slice_tab.addTab(self.canvas_xy, "XY平面")
        self.slice_tab.addTab(self.canvas_xz, "XZ平面")
        self.slice_tab.addTab(self.canvas_yz, "YZ平面")
        right_layout.addWidget(self.slice_tab, stretch=1)

        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, stretch=1)

        # --- ステータスバーに座標・API状態を横並び表示 ---
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)

        self.lbl_click_info = QLabel("")
        self.lbl_click_info.setStyleSheet(
            "font-size: 11px; color: #0077b6; padding: 0 12px; letter-spacing: 1px;"
        )
        sb.addWidget(self.lbl_click_info, 1)

        api_status = "◈  CONNECTED" if self.air else "◇  OFFLINE"
        self.lbl_api_status = QLabel(api_status)
        self.lbl_api_status.setStyleSheet(
            f"font-size: 11px; padding: 0 12px; letter-spacing: 1px; "
            f"color: {'#00875a' if self.air else '#d32f2f'};"
        )
        sb.addPermanentWidget(self.lbl_api_status)

        self.measure_mode = False

    def _make_voxel_spin(self, layout, label, row):
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setRange(0.001, 100.0)
        spin.setDecimals(4)
        spin.setValue(0.1)
        spin.setSingleStep(0.01)
        layout.addWidget(lbl, row, 0)
        layout.addWidget(spin, row, 1)
        return spin

    # -------------------------------------------------------------------------
    # シグナル接続
    # -------------------------------------------------------------------------
    def _connect_signals(self):
        self.btn_load_from_air.clicked.connect(self._on_load_from_air)
        self.btn_load_series.clicked.connect(self._on_load_series)
        self.btn_load_file.clicked.connect(self._on_load_file)
        self.btn_reconnect.clicked.connect(self._on_reconnect)
        self.btn_redraw_volume.clicked.connect(self._on_redraw_volume)
        self.btn_load_xyz.clicked.connect(self._on_load_xyz)
        self.btn_apply_voxel.clicked.connect(self._on_apply_voxel)

        self.slider_z.valueChanged.connect(self._on_slider_z_changed)
        self.spin_z.valueChanged.connect(self._on_spin_z_changed)

        self.slider_y.valueChanged.connect(self._on_slider_y_changed)
        self.spin_y.valueChanged.connect(self._on_spin_y_changed)

        self.slider_x.valueChanged.connect(self._on_slider_x_changed)
        self.spin_x.valueChanged.connect(self._on_spin_x_changed)

        self.combo_cmap.currentTextChanged.connect(self._update_all_slices)
        self.chk_auto_intensity.stateChanged.connect(self._on_auto_intensity_changed)
        self.spin_pct_high.valueChanged.connect(self._on_pct_high_changed)
        self.spin_pct_low.valueChanged.connect(self._on_pct_low_changed)
        self.spin_vmin.valueChanged.connect(self._update_all_slices)
        self.spin_vmax.valueChanged.connect(self._update_all_slices)
        self.btn_intensity_from_slice.clicked.connect(self._on_intensity_from_slice)
        self.btn_intensity_from_volume.clicked.connect(self._on_intensity_from_volume)
        self.chk_show_atoms.stateChanged.connect(self._update_atom_overlay)
        self.slice_tab.currentChanged.connect(self._on_plane_tab_changed)
        self.chk_cluster_link_hologram.toggled.connect(self._on_cluster_link_changed)
        for _ax in ('x', 'y', 'z'):
            self.slider_cluster[_ax].valueChanged.connect(
                lambda idx, a=_ax: self._on_cluster_slider_changed(a, idx))
            self.spin_cluster_pos[_ax].valueChanged.connect(
                lambda ang, a=_ax: self._on_cluster_spin_axis_changed(a, ang))
        self.btn_cluster_sync.clicked.connect(self._on_cluster_sync)

        self.btn_toggle_measure.toggled.connect(self._toggle_measure_mode)
        self.btn_clear_measure.clicked.connect(self._clear_current_tab)
        self.btn_reset_dist.clicked.connect(self._on_reset_dist)
        self.btn_reset_angle.clicked.connect(self._on_reset_angle)
        self.btn_export_dist_csv.clicked.connect(self._export_distance_csv)
        self.btn_export_angle_csv.clicked.connect(self._export_angle_csv)
        self.btn_export_peaks_csv.clicked.connect(self._export_peaks_csv)
        self.spin_fit_size.valueChanged.connect(self._on_fit_size_changed)
        self.btn_grp_measure_type.buttonClicked.connect(self._on_measure_type_changed)
        self.btn_detect_peaks.clicked.connect(self._on_detect_peaks)
        self.chk_show_peak_markers.stateChanged.connect(self._on_show_peaks_changed)
        self.radio_peak_zrange.toggled.connect(self._on_peak_range_changed)
        self.chk_freeze_markers.stateChanged.connect(lambda _: self._redraw_measurements())

        self.btn_remove_xyz.clicked.connect(self._on_remove_xyz)
        self.btn_show_all_xyz.clicked.connect(self._on_show_all_xyz)
        self.btn_hide_all_xyz.clicked.connect(self._on_hide_all_xyz)
        self.list_xyz_files.itemChanged.connect(self._on_xyz_list_item_changed)
        self.list_xyz_files.currentRowChanged.connect(self._on_xyz_list_selection_changed)

        self.spin_offset_x.valueChanged.connect(lambda v: self._on_offset_changed('x', v))
        self.spin_offset_y.valueChanged.connect(lambda v: self._on_offset_changed('y', v))
        self.spin_offset_z.valueChanged.connect(lambda v: self._on_offset_changed('z', v))
        self.btn_reset_offset.clicked.connect(self._on_reset_offset)

        self.canvas_xy.point_clicked.connect(
            lambda x, y: self._on_canvas_click(x, y, plane='xy'))
        self.canvas_xz.point_clicked.connect(
            lambda x, y: self._on_canvas_click(x, y, plane='xz'))
        self.canvas_yz.point_clicked.connect(
            lambda x, y: self._on_canvas_click(x, y, plane='yz'))

    # -------------------------------------------------------------------------
    # API再接続
    # -------------------------------------------------------------------------
    def _on_reconnect(self):
        global air_api
        air_api = None
        self.air = connect_air_api()
        if self.air:
            self.btn_load_from_air.setEnabled(True)
            self.btn_load_series.setEnabled(True)
            self.lbl_api_status.setText("◈  CONNECTED")
            self.lbl_api_status.setStyleSheet("font-size: 11px; padding: 0 12px; letter-spacing: 1px; color: #00875a;")
            self.statusBar().showMessage("3D-AIR-IMAGE API 再接続成功")
        else:
            self.btn_load_from_air.setEnabled(False)
            self.btn_load_series.setEnabled(False)
            self.lbl_api_status.setText("◇  OFFLINE")
            self.lbl_api_status.setStyleSheet("font-size: 11px; padding: 0 12px; letter-spacing: 1px; color: #d32f2f;")
            QMessageBox.warning(self, "接続失敗",
                "3D-AIR-IMAGE API に接続できませんでした。\n"
                "3D-AIR-IMAGEが起動していることを確認してください。")

    # -------------------------------------------------------------------------
    # ボリューム表示再描画 (データ再取得なし)
    # -------------------------------------------------------------------------
    def _on_redraw_volume(self):
        """描画キャッシュクリア + ビューポートリセット + 3平面再描画。
        データ・測定履歴は一切変更しない。"""
        if self.volume is None:
            self.statusBar().showMessage("ボリュームデータが未読み込みです")
            return

        # 各キャンバスのビューポートをextentに合わせてリセット
        for plane, canvas in [('xy', self.canvas_xy), ('xz', self.canvas_xz), ('yz', self.canvas_yz)]:
            ext = self._get_extent(plane)
            canvas.ax.set_xlim(ext[0], ext[1])
            canvas.ax.set_ylim(ext[2], ext[3])

        self._update_all_slices()
        self.statusBar().showMessage("ボリュームを再描画しました")

    # -------------------------------------------------------------------------
    # 3D-AIR-IMAGE から取得
    # -------------------------------------------------------------------------
    def _on_load_from_air(self):
        """ウィンドウを選択してimage_getでデータ取得"""
        if not self.air:
            QMessageBox.warning(self, "エラー", "APIに接続されていません")
            return

        dialog = AirWindowSelectDialog(self, self.air, mode="volume")
        if dialog.exec_() != QDialog.Accepted:
            return

        wid = dialog.selected_wid
        caption = dialog.selected_caption

        try:
            self.statusBar().showMessage(f"データ取得中: {caption} ...")
            QApplication.processEvents()

            # シリーズかどうか確認
            is_series = False
            try:
                is_series = self.air.window_is_mode_series_get(wid)
            except Exception:
                pass

            if is_series:
                self._load_volume_from_series(wid, caption)
            else:
                # 単一ウィンドウ — image_get で取得
                img = self.air.image_get(wid).astype(np.float64)

                if img.ndim == 3:
                    volume = img
                elif img.ndim == 2:
                    QMessageBox.information(self, "情報",
                        f"2D画像です ({img.shape[1]}×{img.shape[0]})。\n"
                        "3Dボリュームが必要な場合は「Series → 3Dボリューム構築」を使用してください。\n"
                        "1スライスとして表示します。")
                    volume = img[np.newaxis, :, :]
                else:
                    raise ValueError(f"予期しないデータ次元: {img.ndim}D")

                self._set_volume_data(volume, caption)

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"取得失敗:\n{str(e)}")

    # -------------------------------------------------------------------------
    # Series → 3Dボリューム構築
    # -------------------------------------------------------------------------
    def _on_load_series(self):
        if not self.air:
            QMessageBox.warning(self, "エラー", "APIに接続されていません")
            return

        dialog = AirWindowSelectDialog(self, self.air, mode="series")
        if dialog.exec_() != QDialog.Accepted:
            return

        self._load_volume_from_series(dialog.selected_wid, dialog.selected_caption)

    def _load_volume_from_series(self, wid, caption):
        """シリーズの全フレームを積み上げて3Dボリュームを構築"""
        try:
            count = self.air.series_get_count(wid)
            if count <= 0:
                QMessageBox.warning(self, "警告", "シリーズにフレームがありません")
                return

            self.statusBar().showMessage(f"ボリューム構築中: {count} フレーム...")
            QApplication.processEvents()

            slices = []
            for i in range(count):
                self.air.series_imageindex_set(wid, i)
                img = self.air.image_get(wid).astype(np.float64)
                slices.append(img)

                if (i + 1) % 10 == 0 or i == count - 1:
                    self.statusBar().showMessage(
                        f"フレーム読み込み中: {i+1}/{count}")
                    QApplication.processEvents()

            volume = np.stack(slices, axis=0)  # (Nz, Ny, Nx)
            self._set_volume_data(volume, f"{caption} (Series: {count} frames)")

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"シリーズ読み込み失敗:\n{str(e)}")

    # -------------------------------------------------------------------------
    # ファイルから読み込み (.xml_air 対応)
    # -------------------------------------------------------------------------
    def _on_load_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "ボリュームデータを選択",
            "", "Volume files (*.npy *.npz *.xml_air);;All files (*)")
        if not filepath:
            return
        try:
            ext = Path(filepath).suffix.lower()
            if ext == '.npy':
                volume = np.load(filepath)
            elif ext == '.npz':
                data = np.load(filepath)
                volume = data[list(data.keys())[0]]
            elif ext == '.xml_air':
                # .xml_air形式の読み込み
                volume, metadata = self._load_xml_air(filepath)
                if metadata:
                    # 算出したボクセルサイズをUIのSpinBoxに反映
                    self.spin_dx.setValue(metadata.get('dx', 0.1))
                    self.spin_dy.setValue(metadata.get('dy', 0.1))
                    self.spin_dz.setValue(metadata.get('dz', 0.1))
            else:
                raise ValueError(f"未対応の形式: {ext}")

            if volume.ndim != 3:
                raise ValueError(f"3Dデータが必要です（現在: {volume.ndim}D）")

            self._set_volume_data(volume, Path(filepath).name)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"読み込み失敗:\n{str(e)}")

    # -------------------------------------------------------------------------
    # .xml_air 専用の読み込み処理
    # -------------------------------------------------------------------------
    def _load_xml_air(self, filepath):
        import xml.etree.ElementTree as ET
        import numpy as np

        tree = ET.parse(filepath)
        root = tree.getroot()

        # 1. XMLから軸情報(Min/Max)の取得
        axis_info = {}
        for i_tag in root.findall('.//InfoValue/i'):
            key_tag = i_tag.find('key')
            val_tag = i_tag.find('val')
            if key_tag is not None and val_tag is not None:
                try:
                    axis_info[key_tag.text] = float(val_tag.text)
                except ValueError:
                    pass

        x_min = axis_info.get('__Axis.X.Min', -10.0)
        x_max = axis_info.get('__Axis.X.Max', 10.0)
        y_min = axis_info.get('__Axis.Y.Min', -10.0)
        y_max = axis_info.get('__Axis.Y.Max', 10.0)
        z_min = axis_info.get('__Axis.Z.Min', -10.0)
        z_max = axis_info.get('__Axis.Z.Max', 10.0)

        # 2. 巨大データ配列の抽出
        # どこかのタグ内に長大なスペース区切りのテキストとしてデータが格納されているため、
        # 最も文字数が多いテキストブロックを抽出する
        longest_text = ""
        for elem in root.iter():
            if elem.text and len(elem.text) > len(longest_text):
                longest_text = elem.text

        if not longest_text or len(longest_text) < 1000:
            raise ValueError("XML内にボリュームデータの配列が見つかりませんでした。")

        # Numpyで高速テキストパース (スペース区切り想定)
        volume_1d = np.fromstring(longest_text, sep=' ', dtype=np.float32)
        if volume_1d.size == 0:
            # カンマ区切りの可能性へのフォールバック
            volume_1d = np.fromstring(longest_text, sep=',', dtype=np.float32)
        
        if volume_1d.size == 0:
            raise ValueError("データのパースに失敗しました（数値データではありません）。")

        # 3. 3D形状へのリシェイプ
        total_size = volume_1d.size
        # 一般的な3D-AIR-IMAGEの出力に合わせて、完全な立方体（N^3）であると仮定してNを逆算
        n = int(np.round(total_size ** (1/3.0)))
        
        if n**3 != total_size:
            raise ValueError(f"データ数 ({total_size}) が立方体のサイズ(N^3)と一致しません。非対称な解像度です。")

        nx = ny = nz = n
        
        # (Z, Y, X) の順でNumPyの3D配列に変換（スライダーと連携させるため）
        volume = volume_1d.reshape((nz, ny, nx))

        # 4. ボクセルサイズ (dx, dy, dz) の算出
        dx = (x_max - x_min) / max(1, nx - 1)
        dy = (y_max - y_min) / max(1, ny - 1)
        dz = (z_max - z_min) / max(1, nz - 1)

        metadata = {'dx': abs(dx), 'dy': abs(dy), 'dz': abs(dz)}
        return volume, metadata

    # -------------------------------------------------------------------------
    # ボリュームデータ設定（共通処理）
    # -------------------------------------------------------------------------
    def _set_volume_data(self, volume, source_name=""):
        self.volume = volume.astype(np.float64)
        nz, ny, nx = self.volume.shape

        dx = self.spin_dx.value()
        dy = self.spin_dy.value()
        dz = self.spin_dz.value()

        # X/Y/Z すべて中心原点（ボリューム中心 = 0 Å）
        self.metadata = {
            'voxel_size': (dz, dy, dx),
            'origin': (0.0, 0.0, 0.0),
            'x_range': (-nx * dx / 2, nx * dx / 2),
            'y_range': (-ny * dy / 2, ny * dy / 2),
            'z_range': (-nz * dz / 2, nz * dz / 2),
        }

        self.slider_z.setRange(0, nz - 1)
        self.spin_z.setRange(self.metadata['z_range'][0], self.metadata['z_range'][1])
        self.spin_z.setSingleStep(dz)
        self.spin_z.blockSignals(True)
        self.spin_z.setValue(0.0)
        self.spin_z.blockSignals(False)
        self.slider_z.blockSignals(True)
        self.slider_z.setValue(nz // 2)
        self.slider_z.blockSignals(False)

        self.slider_y.setRange(0, ny - 1)
        self.spin_y.setRange(self.metadata['y_range'][0], self.metadata['y_range'][1])
        self.spin_y.setSingleStep(dy)
        self.spin_y.blockSignals(True)
        self.spin_y.setValue(0.0)
        self.spin_y.blockSignals(False)
        self.slider_y.blockSignals(True)
        self.slider_y.setValue(ny // 2)
        self.slider_y.blockSignals(False)

        self.slider_x.setRange(0, nx - 1)
        self.spin_x.setRange(self.metadata['x_range'][0], self.metadata['x_range'][1])
        self.spin_x.setSingleStep(dx)
        self.spin_x.blockSignals(True)
        self.spin_x.setValue(0.0)
        self.spin_x.blockSignals(False)
        self.slider_x.blockSignals(True)
        self.slider_x.setValue(nx // 2)
        self.slider_x.blockSignals(False)

        for _axis, _n, _d, _rk in [
                ('z', nz, dz, 'z_range'), ('y', ny, dy, 'y_range'), ('x', nx, dx, 'x_range')]:
            _sp = self.spin_cluster_pos[_axis]
            _sp.blockSignals(True)
            _sp.setRange(self.metadata[_rk][0], self.metadata[_rk][1])
            _sp.setSingleStep(_d)
            _sp.setValue(0.0)
            _sp.blockSignals(False)
            _sl = self.slider_cluster[_axis]
            _sl.blockSignals(True)
            _sl.setRange(0, _n - 1)
            _sl.setValue(_n // 2)
            _sl.blockSignals(False)

        self.lbl_volume_info.setText(
            f"サイズ: {nx}×{ny}×{nz}\n"
            f"値域: [{volume.min():.3g}, {volume.max():.3g}]\n"
            f"ソース: {source_name}")

        self._update_all_slices()
        self.statusBar().showMessage(
            f"ボリューム読み込み完了: {nx}×{ny}×{nz} — {source_name}")

    # -------------------------------------------------------------------------
    # ボクセルサイズ適用
    # -------------------------------------------------------------------------
    def _on_apply_voxel(self):
        if self.volume is None:
            return
        dx, dy, dz = self.spin_dx.value(), self.spin_dy.value(), self.spin_dz.value()
        nz, ny, nx = self.volume.shape
        self.metadata['voxel_size'] = (dz, dy, dx)
        self.metadata['x_range'] = (-nx * dx / 2, nx * dx / 2)
        self.metadata['y_range'] = (-ny * dy / 2, ny * dy / 2)
        self.metadata['z_range'] = (-nz * dz / 2, nz * dz / 2)
        self._update_all_slices()
        self.statusBar().showMessage(f"ボクセルサイズ更新: dx={dx}, dy={dy}, dz={dz} Å")

    # -------------------------------------------------------------------------
    # XYZ読み込み・管理
    # -------------------------------------------------------------------------
    def _on_load_xyz(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "XYZファイルを選択 (複数可)", "", "XYZ files (*.xyz);;All files (*)")
        if not filepaths:
            return
        added = 0
        for fp in filepaths:
            p = Path(fp)
            if any(s['filepath'] == p for s in self.atom_sets):
                continue
            try:
                atoms = load_xyz(str(p))
                if not atoms:
                    continue
                elements = set(a['element'] for a in atoms)
                atom_set = {
                    'filepath': p,
                    'name': p.stem,
                    'atoms': atoms,
                    'visible': True,
                    'offset': [0.0, 0.0, 0.0],
                    'element_colors': {e: get_element_color(e) for e in elements},
                    'element_radii': {e: get_element_radius(e) for e in elements},
                }
                self.atom_sets.append(atom_set)
                added += 1
            except Exception as e:
                self.statusBar().showMessage(f"読み込み失敗: {p.name}: {e}")
        if added > 0:
            self._rebuild_xyz_list_ui()
            self.list_xyz_files.setCurrentRow(len(self.atom_sets) - 1)
            self._update_atom_overlay()
            self.statusBar().showMessage(f"XYZ読み込み完了: {added}ファイル追加")
        self._update_xyz_info_label()

    def _rebuild_xyz_list_ui(self):
        """atom_sets の内容に基づいて list_xyz_files を再構築する。"""
        self.list_xyz_files.blockSignals(True)
        self.list_xyz_files.clear()
        for s in self.atom_sets:
            item = QListWidgetItem(s['name'])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s['visible'] else Qt.Unchecked)
            self.list_xyz_files.addItem(item)
        self.list_xyz_files.blockSignals(False)

    def _on_xyz_list_item_changed(self, item):
        """チェックボックス変更: visible を更新して再描画。"""
        idx = self.list_xyz_files.row(item)
        if 0 <= idx < len(self.atom_sets):
            self.atom_sets[idx]['visible'] = (item.checkState() == Qt.Checked)
            self._update_atom_overlay()
            self._update_xyz_info_label()

    def _on_xyz_list_selection_changed(self, row):
        """リスト行選択: active_atom_set_index を更新して Element/Offset UI を切替。"""
        if 0 <= row < len(self.atom_sets):
            self.active_atom_set_index = row
        else:
            self.active_atom_set_index = -1
        self._update_active_ui()

    def _on_remove_xyz(self):
        """選択中の XYZ セットを削除する。"""
        row = self.list_xyz_files.currentRow()
        if row < 0 or row >= len(self.atom_sets):
            return
        del self.atom_sets[row]
        if self.active_atom_set_index >= len(self.atom_sets):
            self.active_atom_set_index = len(self.atom_sets) - 1
        self._rebuild_xyz_list_ui()
        if self.active_atom_set_index >= 0:
            self.list_xyz_files.setCurrentRow(self.active_atom_set_index)
        self._update_active_ui()
        self._update_atom_overlay()
        self._update_xyz_info_label()

    def _on_show_all_xyz(self):
        for s in self.atom_sets:
            s['visible'] = True
        self._rebuild_xyz_list_ui()
        self._update_atom_overlay()
        self._update_xyz_info_label()

    def _on_hide_all_xyz(self):
        for s in self.atom_sets:
            s['visible'] = False
        self._rebuild_xyz_list_ui()
        self._update_atom_overlay()
        self._update_xyz_info_label()

    def _update_xyz_info_label(self):
        total = len(self.atom_sets)
        visible = sum(1 for s in self.atom_sets if s['visible'])
        self.lbl_xyz_info.setText(f"読み込み済み: {total} / 表示中: {visible}")

    def _update_active_ui(self):
        """active_atom_set_index に基づいて Element UI と Offset UI を更新する。"""
        has_active = 0 <= self.active_atom_set_index < len(self.atom_sets)
        name = self.atom_sets[self.active_atom_set_index]['name'] if has_active else "—"
        self.lbl_active_set_name.setText(f"編集中: {name}")
        self.lbl_active_elem_set.setText(f"編集中: {name}")
        self.btn_reset_offset.setEnabled(has_active)
        self._update_offset_axis_enabled()
        if has_active:
            off = self.atom_sets[self.active_atom_set_index]['offset']
            self.spin_offset_x.blockSignals(True)
            self.spin_offset_y.blockSignals(True)
            self.spin_offset_z.blockSignals(True)
            self.spin_offset_x.setValue(off[0])
            self.spin_offset_y.setValue(off[1])
            self.spin_offset_z.setValue(off[2])
            self.spin_offset_x.blockSignals(False)
            self.spin_offset_y.blockSignals(False)
            self.spin_offset_z.blockSignals(False)
        self._rebuild_element_settings_ui()

    def _on_offset_changed(self, axis, value):
        if not (0 <= self.active_atom_set_index < len(self.atom_sets)):
            return
        idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        self.atom_sets[self.active_atom_set_index]['offset'][idx] = value
        self._update_atom_overlay()

    def _on_reset_offset(self):
        if not (0 <= self.active_atom_set_index < len(self.atom_sets)):
            return
        self.atom_sets[self.active_atom_set_index]['offset'] = [0.0, 0.0, 0.0]
        self.spin_offset_x.blockSignals(True)
        self.spin_offset_y.blockSignals(True)
        self.spin_offset_z.blockSignals(True)
        self.spin_offset_x.setValue(0.0)
        self.spin_offset_y.setValue(0.0)
        self.spin_offset_z.setValue(0.0)
        self.spin_offset_x.blockSignals(False)
        self.spin_offset_y.blockSignals(False)
        self.spin_offset_z.blockSignals(False)
        self._update_atom_overlay()

    # -------------------------------------------------------------------------
    # スライス更新
    # -------------------------------------------------------------------------
    def _get_extent(self, plane):
        m = self.metadata
        if plane == 'xy':
            return [m['x_range'][0], m['x_range'][1], m['y_range'][0], m['y_range'][1]]
        elif plane == 'xz':
            return [m['x_range'][0], m['x_range'][1], m['z_range'][0], m['z_range'][1]]
        elif plane == 'yz':
            return [m['y_range'][0], m['y_range'][1], m['z_range'][0], m['z_range'][1]]

    def _update_slice(self):
        """後方互換エイリアス。XYスライスのみ更新。"""
        self._update_slice_xy()

    def _update_slice_xy(self):
        """XY平面 (Z固定) スライスを再描画する。"""
        if self.volume is None:
            return
        cmap = self.combo_cmap.currentText()
        z_pos = self.spin_z.value()
        idx = self._angstrom_to_slice_idx('z', z_pos)
        data = self.volume[idx, :, :]
        extent = self._get_extent('xy')
        self.canvas_xy.ax.set_xlabel("X (Å)", color='#4a6880', fontsize=11)
        self.canvas_xy.ax.set_ylabel("Y (Å)", color='#4a6880', fontsize=11)
        self.canvas_xy.ax.set_title(
            f"XY  ·  Z = {z_pos:+.3f} Å",
            color='#0077b6', fontsize=13, fontweight='bold')
        vmin, vmax = self._compute_vmin_vmax(data)
        self.canvas_xy.display_slice(data, extent, cmap, vmin=vmin, vmax=vmax)
        if self.chk_cluster_link_hologram.isChecked():
            self.spin_cluster_pos['z'].blockSignals(True)
            self.spin_cluster_pos['z'].setValue(z_pos)
            self.spin_cluster_pos['z'].blockSignals(False)
        self._update_atom_overlay_for_plane('xy')
        self._redraw_measurements_on_canvas('xy')
        self._update_peak_display_on_canvas('xy')

    def _update_slice_xz(self):
        """XZ平面 (Y固定) スライスを再描画する。"""
        if self.volume is None:
            return
        cmap = self.combo_cmap.currentText()
        y_pos = self.spin_y.value()
        iy = self._angstrom_to_slice_idx('y', y_pos)
        data = self.volume[:, iy, :]   # shape (Nz, Nx); rows=Z, cols=X
        extent = self._get_extent('xz')
        self.canvas_xz.ax.set_xlabel("X (Å)", color='#4a6880', fontsize=11)
        self.canvas_xz.ax.set_ylabel("Z (Å)", color='#4a6880', fontsize=11)
        self.canvas_xz.ax.set_title(
            f"XZ  ·  Y = {y_pos:+.3f} Å",
            color='#0077b6', fontsize=13, fontweight='bold')
        vmin, vmax = self._compute_vmin_vmax(data)
        self.canvas_xz.display_slice(data, extent, cmap, vmin=vmin, vmax=vmax)
        if self.chk_cluster_link_hologram.isChecked():
            self.spin_cluster_pos['y'].blockSignals(True)
            self.spin_cluster_pos['y'].setValue(y_pos)
            self.spin_cluster_pos['y'].blockSignals(False)
        self._update_atom_overlay_for_plane('xz')
        self._redraw_measurements_on_canvas('xz')
        self._update_peak_display_on_canvas('xz')

    def _update_slice_yz(self):
        """YZ平面 (X固定) スライスを再描画する。"""
        if self.volume is None:
            return
        cmap = self.combo_cmap.currentText()
        x_pos = self.spin_x.value()
        ix = self._angstrom_to_slice_idx('x', x_pos)
        data = self.volume[:, :, ix]   # shape (Nz, Ny); rows=Z, cols=Y
        extent = self._get_extent('yz')
        self.canvas_yz.ax.set_xlabel("Y (Å)", color='#4a6880', fontsize=11)
        self.canvas_yz.ax.set_ylabel("Z (Å)", color='#4a6880', fontsize=11)
        self.canvas_yz.ax.set_title(
            f"YZ  ·  X = {x_pos:+.3f} Å",
            color='#0077b6', fontsize=13, fontweight='bold')
        vmin, vmax = self._compute_vmin_vmax(data)
        self.canvas_yz.display_slice(data, extent, cmap, vmin=vmin, vmax=vmax)
        if self.chk_cluster_link_hologram.isChecked():
            self.spin_cluster_pos['x'].blockSignals(True)
            self.spin_cluster_pos['x'].setValue(x_pos)
            self.spin_cluster_pos['x'].blockSignals(False)
        self._update_atom_overlay_for_plane('yz')
        self._redraw_measurements_on_canvas('yz')
        self._update_peak_display_on_canvas('yz')

    def _update_all_slices(self):
        """3平面すべてを更新する。"""
        self._update_slice_xy()
        self._update_slice_xz()
        self._update_slice_yz()

    # -------------------------------------------------------------------------
    # 強度調整
    # -------------------------------------------------------------------------
    def _compute_vmin_vmax(self, data_2d):
        """現在の設定から vmin/vmax を返す"""
        if self.chk_auto_intensity.isChecked():
            plo = self.spin_pct_low.value()
            phi = self.spin_pct_high.value()
            return float(np.percentile(data_2d, plo)), float(np.percentile(data_2d, phi))
        else:
            return self.spin_vmin.value(), self.spin_vmax.value()

    def _on_auto_intensity_changed(self, state):
        auto = bool(state)
        self.spin_pct_low.setEnabled(auto)
        self.spin_pct_high.setEnabled(auto)
        self.spin_vmin.setEnabled(not auto)
        self.spin_vmax.setEnabled(not auto)
        self._update_all_slices()

    def _on_pct_high_changed(self, value):
        """上限が変わったら下限の最大値を上限-1に制限して再描画"""
        self.spin_pct_low.setMaximum(round(value - 1.0, 1))
        self._update_all_slices()

    def _on_pct_low_changed(self, value):
        """下限が変わったら上限の最小値を下限+1に制限して再描画"""
        self.spin_pct_high.setMinimum(round(value + 1.0, 1))
        self._update_all_slices()

    def _on_intensity_from_slice(self):
        if self.volume is None:
            return
        data = self.volume[self._angstrom_to_slice_idx('z', self.spin_z.value()), :, :]
        self.chk_auto_intensity.setChecked(False)
        self.spin_vmin.blockSignals(True)
        self.spin_vmax.blockSignals(True)
        self.spin_vmin.setValue(float(data.min()))
        self.spin_vmax.setValue(float(data.max()))
        self.spin_vmin.blockSignals(False)
        self.spin_vmax.blockSignals(False)
        self._update_all_slices()

    def _on_intensity_from_volume(self):
        if self.volume is None:
            return
        self.chk_auto_intensity.setChecked(False)
        self.spin_vmin.blockSignals(True)
        self.spin_vmax.blockSignals(True)
        self.spin_vmin.setValue(float(self.volume.min()))
        self.spin_vmax.setValue(float(self.volume.max()))
        self.spin_vmin.blockSignals(False)
        self.spin_vmax.blockSignals(False)
        self._update_all_slices()

    @property
    def atoms(self):
        """後方互換: アクティブセットの atoms を返す (なければ空リスト)。"""
        if 0 <= self.active_atom_set_index < len(self.atom_sets):
            return self.atom_sets[self.active_atom_set_index]['atoms']
        return []

    # 原子オーバーレイ
    # -------------------------------------------------------------------------
    def _current_slice_pos_angstrom(self, axis: str) -> float:
        """現在のスライス位置を Å 単位で返す (軸: 'x'/'y'/'z')。spin_z/y/x が Å 値を保持。"""
        return {'z': self.spin_z, 'y': self.spin_y, 'x': self.spin_x}[axis].value()

    def _angstrom_to_slice_idx(self, axis: str, ang: float) -> int:
        """Å座標を最寄りスライスインデックスに変換する。"""
        d = self.metadata['voxel_size'][{'z': 0, 'y': 1, 'x': 2}[axis]]
        n = self.volume.shape[{'z': 0, 'y': 1, 'x': 2}[axis]]
        return max(0, min(n - 1, int(round(ang / d)) + n // 2))

    def _current_cluster_pos_angstrom(self, axis: str) -> float:
        """原子表示の基準位置を Å で返す。連動モードならホログラム位置、独立モードなら専用スピン値。"""
        if self.chk_cluster_link_hologram.isChecked():
            return self._current_slice_pos_angstrom(axis)
        return self.spin_cluster_pos[axis].value()

    def _on_plane_tab_changed(self, _idx):
        """平面タブ切り替え時にオフセット軸の有効/無効を更新する。"""
        self._update_offset_axis_enabled()

    def _update_offset_axis_enabled(self):
        """現在の平面タブと active_atom_set に応じてオフセット軸を有効/無効化する。
        法線方向の軸のみ有効 (XY→Z, XZ→Y, YZ→X)。"""
        has_active = 0 <= self.active_atom_set_index < len(self.atom_sets)
        tab = self.slice_tab.currentIndex()  # 0=XY, 1=XZ, 2=YZ
        self.spin_offset_x.setEnabled(has_active and tab == 2)
        self.spin_offset_y.setEnabled(has_active and tab == 1)
        self.spin_offset_z.setEnabled(has_active and tab == 0)

    def _update_atom_overlay(self):
        """3平面すべての原子オーバーレイを更新する。"""
        for plane in ('xy', 'xz', 'yz'):
            self._update_atom_overlay_for_plane(plane)

    def _update_atom_overlay_for_plane(self, plane='xy'):
        """指定平面のキャンバスに原子オーバーレイを描画する。
        判定: 法線方向座標が |Δ| ≤ voxel_size/2 の原子のみ表示。
        描画座標: 面内2軸のみ使用。法線方向オフセットは判定にのみ使う。"""
        canvas = self._get_canvas(plane)
        if self.volume is None:
            return
        canvas.clear_atoms()

        if not self.chk_show_atoms.isChecked() or not self.atom_sets:
            return

        m = self.metadata
        positions, colors, radii = [], [], []

        if plane == 'xy':
            z_slice = self._current_cluster_pos_angstrom('z')
            dz = m['voxel_size'][0]
            for atom_set in self.atom_sets:
                if not atom_set['visible']:
                    continue
                ox_off, oy_off, oz_off = atom_set['offset']
                ec, er = atom_set['element_colors'], atom_set['element_radii']
                for a in atom_set['atoms']:
                    if abs((a['z'] + oz_off) - z_slice) <= dz / 2:
                        elem = a['element']
                        positions.append((a['x'] + ox_off, a['y'] + oy_off))
                        colors.append(ec.get(elem, get_element_color(elem)))
                        radii.append(er.get(elem, get_element_radius(elem)))

        elif plane == 'xz':
            y_slice = self._current_cluster_pos_angstrom('y')
            dy = m['voxel_size'][1]
            for atom_set in self.atom_sets:
                if not atom_set['visible']:
                    continue
                ox_off, oy_off, oz_off = atom_set['offset']
                ec, er = atom_set['element_colors'], atom_set['element_radii']
                for a in atom_set['atoms']:
                    if abs((a['y'] + oy_off) - y_slice) <= dy / 2:
                        elem = a['element']
                        positions.append((a['x'] + ox_off, a['z'] + oz_off))
                        colors.append(ec.get(elem, get_element_color(elem)))
                        radii.append(er.get(elem, get_element_radius(elem)))

        elif plane == 'yz':
            x_slice = self._current_cluster_pos_angstrom('x')
            dx = m['voxel_size'][2]
            for atom_set in self.atom_sets:
                if not atom_set['visible']:
                    continue
                ox_off, oy_off, oz_off = atom_set['offset']
                ec, er = atom_set['element_colors'], atom_set['element_radii']
                for a in atom_set['atoms']:
                    if abs((a['x'] + ox_off) - x_slice) <= dx / 2:
                        elem = a['element']
                        positions.append((a['y'] + oy_off, a['z'] + oz_off))
                        colors.append(ec.get(elem, get_element_color(elem)))
                        radii.append(er.get(elem, get_element_radius(elem)))

        if positions:
            canvas.overlay_atoms(positions, colors, radii)

    def _rebuild_element_settings_ui(self):
        """アクティブセットの元素ごとの色・半径設定行を再構築する。"""
        while self.elem_rows_layout.count():
            item = self.elem_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not (0 <= self.active_atom_set_index < len(self.atom_sets)):
            self.elem_rows_layout.addWidget(QLabel("XYZファイルを選択してください"))
            return

        atom_set = self.atom_sets[self.active_atom_set_index]
        ec = atom_set['element_colors']
        er = atom_set['element_radii']
        elements = sorted(set(a['element'] for a in atom_set['atoms']))
        if not elements:
            self.elem_rows_layout.addWidget(QLabel("原子データがありません"))
            return

        for elem in elements:
            if elem not in ec:
                ec[elem] = get_element_color(elem)
            if elem not in er:
                er[elem] = get_element_radius(elem)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 1, 0, 1)
            row_layout.setSpacing(4)

            lbl = QLabel(elem)
            lbl.setFixedWidth(42)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"font-weight: bold; color: {ec[elem]}; font-size: 12px;")
            row_layout.addWidget(lbl)

            btn_color = QPushButton()
            btn_color.setFixedSize(52, 26)
            btn_color.setStyleSheet(
                f"background-color: {ec[elem]}; border: 1px solid #45475a; border-radius: 3px;")

            def _make_color_cb(e, btn, lbl_ref, aset):
                def cb():
                    from PyQt5.QtGui import QColor as _QColor
                    cur = _QColor(aset['element_colors'][e])
                    c = QColorDialog.getColor(cur, self, f"{e} の色")
                    if c.isValid():
                        hex_c = c.name()
                        aset['element_colors'][e] = hex_c
                        btn.setStyleSheet(
                            f"background-color: {hex_c}; border: 1px solid #45475a; border-radius: 3px;")
                        lbl_ref.setStyleSheet(
                            f"font-weight: bold; color: {hex_c}; font-size: 13px;")
                        self._update_atom_overlay()
                return cb

            btn_color.clicked.connect(_make_color_cb(elem, btn_color, lbl, atom_set))
            row_layout.addWidget(btn_color)

            spin_r = QDoubleSpinBox()
            spin_r.setRange(0.01, 20.0)
            spin_r.setDecimals(2)
            spin_r.setValue(er[elem])
            spin_r.setSingleStep(0.1)
            spin_r.setFixedWidth(76)

            def _make_radius_cb(e, aset):
                def cb(val):
                    aset['element_radii'][e] = val
                    self._update_atom_overlay()
                return cb

            spin_r.valueChanged.connect(_make_radius_cb(elem, atom_set))
            row_layout.addWidget(spin_r)
            row_layout.addStretch()

            self.elem_rows_layout.addWidget(row)

        self.elem_rows_layout.addStretch()

    def _on_cluster_link_changed(self, linked: bool):
        """連動/独立モード切替: クラスタースピン/スライダーの有効無効とオーバーレイ更新。"""
        for axis in ('x', 'y', 'z'):
            self.spin_cluster_pos[axis].setEnabled(not linked)
            self.slider_cluster[axis].setEnabled(not linked)
        if linked and self.metadata:
            for axis in ('x', 'y', 'z'):
                ang = self._current_slice_pos_angstrom(axis)
                self.spin_cluster_pos[axis].blockSignals(True)
                self.spin_cluster_pos[axis].setValue(ang)
                self.spin_cluster_pos[axis].blockSignals(False)
                if self.volume is not None:
                    d = self.metadata['voxel_size'][{'z': 0, 'y': 1, 'x': 2}[axis]]
                    n = self.volume.shape[{'z': 0, 'y': 1, 'x': 2}[axis]]
                    self.slider_cluster[axis].blockSignals(True)
                    self.slider_cluster[axis].setValue(
                        max(0, min(n - 1, int(round(ang / d)) + n // 2)))
                    self.slider_cluster[axis].blockSignals(False)
        self._update_atom_overlay()

    def _on_cluster_sync(self):
        """現在のホログラム位置をクラスタースピンにコピーする。"""
        if self.metadata is None:
            return
        for axis in ('x', 'y', 'z'):
            ang = self._current_slice_pos_angstrom(axis)
            self.spin_cluster_pos[axis].blockSignals(True)
            self.spin_cluster_pos[axis].setValue(ang)
            self.spin_cluster_pos[axis].blockSignals(False)
            if self.volume is not None:
                d = self.metadata['voxel_size'][{'z': 0, 'y': 1, 'x': 2}[axis]]
                n = self.volume.shape[{'z': 0, 'y': 1, 'x': 2}[axis]]
                self.slider_cluster[axis].blockSignals(True)
                self.slider_cluster[axis].setValue(
                    max(0, min(n - 1, int(round(ang / d)) + n // 2)))
                self.slider_cluster[axis].blockSignals(False)
        if not self.chk_cluster_link_hologram.isChecked():
            self._update_atom_overlay()

    def _on_cluster_slider_changed(self, axis: str, idx: int):
        """独立モードでクラスタースライダー変更時: スピンへ同期してオーバーレイ更新。"""
        if self.volume is None:
            return
        d = self.metadata['voxel_size'][{'z': 0, 'y': 1, 'x': 2}[axis]]
        n = self.volume.shape[{'z': 0, 'y': 1, 'x': 2}[axis]]
        ang = (idx - n // 2) * d
        self.spin_cluster_pos[axis].blockSignals(True)
        self.spin_cluster_pos[axis].setValue(ang)
        self.spin_cluster_pos[axis].blockSignals(False)
        self._update_atom_overlay()

    def _on_cluster_spin_axis_changed(self, axis: str, ang: float):
        """独立モードでクラスタースピン変更時: スライダーへ同期してオーバーレイ更新。"""
        if self.volume is None:
            return
        d = self.metadata['voxel_size'][{'z': 0, 'y': 1, 'x': 2}[axis]]
        n = self.volume.shape[{'z': 0, 'y': 1, 'x': 2}[axis]]
        self.slider_cluster[axis].blockSignals(True)
        self.slider_cluster[axis].setValue(max(0, min(n - 1, int(round(ang / d)) + n // 2)))
        self.slider_cluster[axis].blockSignals(False)
        self._update_atom_overlay()

    def _on_slider_z_changed(self, idx: int):
        if self.volume is None:
            return
        dz = self.metadata['voxel_size'][0]
        nz = self.volume.shape[0]
        ang = (idx - nz // 2) * dz
        self.spin_z.blockSignals(True)
        self.spin_z.setValue(ang)
        self.spin_z.blockSignals(False)
        self._update_slice_xy()

    def _on_spin_z_changed(self, ang: float):
        if self.volume is None:
            return
        idx = self._angstrom_to_slice_idx('z', ang)
        self.slider_z.blockSignals(True)
        self.slider_z.setValue(idx)
        self.slider_z.blockSignals(False)
        self._update_slice_xy()

    def _on_slider_y_changed(self, idx: int):
        if self.volume is None:
            return
        dy = self.metadata['voxel_size'][1]
        ny = self.volume.shape[1]
        ang = (idx - ny // 2) * dy
        self.spin_y.blockSignals(True)
        self.spin_y.setValue(ang)
        self.spin_y.blockSignals(False)
        self._update_slice_xz()

    def _on_spin_y_changed(self, ang: float):
        if self.volume is None:
            return
        idx = self._angstrom_to_slice_idx('y', ang)
        self.slider_y.blockSignals(True)
        self.slider_y.setValue(idx)
        self.slider_y.blockSignals(False)
        self._update_slice_xz()

    def _on_slider_x_changed(self, idx: int):
        if self.volume is None:
            return
        dx = self.metadata['voxel_size'][2]
        nx = self.volume.shape[2]
        ang = (idx - nx // 2) * dx
        self.spin_x.blockSignals(True)
        self.spin_x.setValue(ang)
        self.spin_x.blockSignals(False)
        self._update_slice_yz()

    def _on_spin_x_changed(self, ang: float):
        if self.volume is None:
            return
        idx = self._angstrom_to_slice_idx('x', ang)
        self.slider_x.blockSignals(True)
        self.slider_x.setValue(idx)
        self.slider_x.blockSignals(False)
        self._update_slice_yz()

    # -------------------------------------------------------------------------
    # 距離測定
    # -------------------------------------------------------------------------
    def _toggle_measure_mode(self, checked):
        self.measure_mode = checked
        if checked:
            self.click_points = []
            self._update_measure_status_label()
        else:
            self.lbl_measure_mode.setText("● MEASURE  OFF")
            self.lbl_measure_mode.setStyleSheet(
                "font-weight: bold; font-size: 11px; letter-spacing: 2px; color: #90a8c0;")

    def _update_measure_status_label(self):
        """Update the MEASURE mode status label to reflect type and progress."""
        if not self.measure_mode:
            return
        if self.measure_type == 'distance':
            self.lbl_measure_mode.setText("◉ MEASURE  ON  — 2点をクリック")
            self.lbl_measure_mode.setStyleSheet(
                "font-weight: bold; font-size: 11px; letter-spacing: 1px; color: #00875a;")
        else:
            n = len(self.click_points)
            self.lbl_measure_mode.setText(
                f"◉ 3点角度モード ON — 3点をクリック (現在 {n}/3)")
            self.lbl_measure_mode.setStyleSheet(
                "font-weight: bold; font-size: 11px; letter-spacing: 1px; color: #e08d3c;")

    def _get_canvas(self, plane):
        """平面名からキャンバスを返すヘルパー。"""
        return {'xy': self.canvas_xy, 'xz': self.canvas_xz, 'yz': self.canvas_yz}[plane]

    def _on_measure_type_changed(self, _btn):
        """測定タイプ変更時: 進行中クリックと全キャンバスのマーカーをクリア。"""
        self.measure_type = 'angle' if self.radio_angle.isChecked() else 'distance'
        self.click_points = []
        for canvas in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
            canvas.clear_canvas_markers()
        self._redraw_measurements()
        if self.measure_mode:
            self._update_measure_status_label()

    def _snap_or_find_peak(self, data_2d, px, py, search_r, x_real, y_real):
        """Return (peak_px, peak_py): snap to nearest detected peak or run _find_peak."""
        if (self.chk_snap_to_peaks.isChecked()
                and self.detected_peaks
                and self.chk_show_peak_markers.isChecked()):
            m = self.metadata
            dx_v, dy_v = m['voxel_size'][2], m['voxel_size'][1]
            ox, oy = m['x_range'][0], m['y_range'][0]
            snap_r_ang = search_r * min(dx_v, dy_v)
            best_d = float('inf')
            best_px, best_py = None, None
            for pk in self.detected_peaks:
                d = np.hypot(pk['x'] - x_real, pk['y'] - y_real)
                if d < snap_r_ang and d < best_d:
                    best_d = d
                    best_px = (pk['x'] - ox) / dx_v
                    best_py = (pk['y'] - oy) / dy_v
            if best_px is not None:
                return best_px, best_py
        return self._find_peak(data_2d, px, py, search_r)

    def _find_peak_3d_refined(self, px, py, z_idx, search_r, x_real, y_real):
        """サブボクセル精緻化付きピーク検索。Returns (x_ang, y_ang, z_ang, log_str) in Å.

        スナップ→粗最大値→選択アルゴリズムの順で処理する。"""
        m = self.metadata
        dx_v, dy_v, dz_v = m['voxel_size'][2], m['voxel_size'][1], m['voxel_size'][0]
        ox, oy, oz = m['x_range'][0], m['y_range'][0], m['z_range'][0]
        data_2d = self.volume[z_idx]

        # スナップチェック (自動検出マーカーへのスナップ)
        if (self.chk_snap_to_peaks.isChecked()
                and self.detected_peaks
                and self.chk_show_peak_markers.isChecked()):
            snap_r_ang = search_r * min(dx_v, dy_v)
            best_d = float('inf')
            best_pk = None
            for pk in self.detected_peaks:
                d = np.hypot(pk['x'] - x_real, pk['y'] - y_real)
                if d < snap_r_ang and d < best_d:
                    best_d = d
                    best_pk = pk
            if best_pk is not None:
                z_pk = best_pk.get('z', oz + z_idx * dz_v)
                return best_pk['x'], best_pk['y'], z_pk, "snap"

        # 粗最大値
        ix_max, iy_max = _peak_find_coarse(data_2d, px, py, search_r)
        coarse_log = f"coarse:({ix_max},{iy_max},{z_idx})"

        refine_mode = self.combo_peak_refine.currentIndex()
        fit_size = self.spin_fit_size.value()

        if refine_mode == PEAK_REFINE_GAUSSIAN:
            result = peak_refine_gaussian_3d(self.volume, z_idx, iy_max, ix_max, fit_size)
            if result is None:
                r = peak_refine_parabola_3d(self.volume, z_idx, iy_max, ix_max)
                x_sub, y_sub, z_sub = r[0], r[1], r[2]
                log = coarse_log + " → gauss_fail→" + r[3]
            else:
                x_sub, y_sub, z_sub = result[0], result[1], result[2]
                log = coarse_log + " → " + result[6]
        elif refine_mode == PEAK_REFINE_CENTROID:
            r = peak_refine_centroid_3d(self.volume, z_idx, iy_max, ix_max, fit_size)
            x_sub, y_sub, z_sub = r[0], r[1], r[2]
            log = coarse_log + " → " + r[3]
        else:  # PEAK_REFINE_PARABOLA
            r = peak_refine_parabola_3d(self.volume, z_idx, iy_max, ix_max)
            x_sub, y_sub, z_sub = r[0], r[1], r[2]
            log = coarse_log + " → " + r[3]

        x_ang = ox + x_sub * dx_v
        y_ang = oy + y_sub * dy_v
        z_ang = oz + z_sub * dz_v

        if self.chk_fit_log.isChecked():
            self.statusBar().showMessage(log)

        return x_ang, y_ang, z_ang, log

    def _on_fit_size_changed(self, val):
        """フィット領域サイズを奇数に強制する。"""
        if val % 2 == 0:
            self.spin_fit_size.blockSignals(True)
            self.spin_fit_size.setValue(val + 1)
            self.spin_fit_size.blockSignals(False)

    def _on_canvas_click(self, x_real, y_real, plane='xy'):
        """canvas クリックを処理する。plane: 'xy'|'xz'|'yz'。"""
        if not self.measure_mode or self.volume is None:
            return

        # 異なる平面のクリックが混在しないようにチェック
        if self.click_points and self.click_points[0].get('plane') != plane:
            self.statusBar().showMessage(
                "警告: 異なる平面の測定点は混在できません。進行中の点をリセットします")
            self.click_points = []
            for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
                cv.clear_canvas_markers()
            self._redraw_measurements()
            return

        m = self.metadata
        dz_v, dy_v, dx_v = m['voxel_size']
        ox, oy, oz = m['x_range'][0], m['y_range'][0], m['z_range'][0]
        search_r = self.spin_search_radius.value()

        if plane == 'xy':
            z_idx = self.slider_z.value()
            data_2d = self.volume[z_idx, :, :]
            # キャンバスは (X, Y) Å; _peak_find_coarse は pixel 空間 (ix, iy)
            px = (x_real - ox) / dx_v
            py = (y_real - oy) / dy_v
            if self.combo_peak_refine.currentIndex() == PEAK_REFINE_OFF:
                peak_px, peak_py = self._snap_or_find_peak(
                    data_2d, px, py, search_r, x_real, y_real)
                peak_x = ox + peak_px * dx_v
                peak_y = oy + peak_py * dy_v
                peak_z = oz + z_idx * dz_v
            else:
                peak_x, peak_y, peak_z, _log = self._find_peak_3d_refined(
                    px, py, z_idx, search_r, x_real, y_real)
                peak_px = (peak_x - ox) / dx_v
                peak_py = (peak_y - oy) / dy_v

        elif plane == 'xz':
            iy = self.slider_y.value()
            # キャンバスは (X, Z) Å; data_2d[iz, ix]
            data_2d = self.volume[:, iy, :]
            px = (x_real - ox) / dx_v   # X pixel
            pz = (y_real - oz) / dz_v   # Z pixel (canvas y-axis = Z)
            # 粗最大値 → 3D精緻化
            ix_max, iz_max = _peak_find_coarse(data_2d, px, pz, search_r)
            refine_mode = self.combo_peak_refine.currentIndex()
            fit_size = self.spin_fit_size.value()
            if refine_mode == PEAK_REFINE_OFF:
                x_sub, y_sub_v, z_sub = float(ix_max), float(iy), float(iz_max)
                log = f"coarse:({ix_max},{iy},{iz_max})"
            else:
                x_sub, y_sub_v, z_sub = _apply_peak_refinement_on_region(
                    self.volume, iz_max, iy, ix_max, refine_mode, fit_size)
                log = f"xz_refine:({x_sub:.3f},{y_sub_v:.3f},{z_sub:.3f})"
            if self.chk_fit_log.isChecked():
                self.statusBar().showMessage(log)
            peak_x = ox + x_sub * dx_v
            peak_y = oy + y_sub_v * dy_v
            peak_z = oz + z_sub * dz_v
            peak_px = x_sub   # X pixel
            peak_py = z_sub   # Z pixel (XZ平面では「py」にZ pixelを格納)

        else:  # plane == 'yz'
            ix = self.slider_x.value()
            # キャンバスは (Y, Z) Å; data_2d[iz, iy]
            data_2d = self.volume[:, :, ix]
            py = (x_real - oy) / dy_v   # Y pixel (canvas x-axis = Y)
            pz = (y_real - oz) / dz_v   # Z pixel
            iy_max, iz_max = _peak_find_coarse(data_2d, py, pz, search_r)
            refine_mode = self.combo_peak_refine.currentIndex()
            fit_size = self.spin_fit_size.value()
            if refine_mode == PEAK_REFINE_OFF:
                x_sub_v, y_sub, z_sub = float(ix), float(iy_max), float(iz_max)
                log = f"coarse:({ix},{iy_max},{iz_max})"
            else:
                x_sub_v, y_sub, z_sub = _apply_peak_refinement_on_region(
                    self.volume, iz_max, iy_max, ix, refine_mode, fit_size)
                log = f"yz_refine:({x_sub_v:.3f},{y_sub:.3f},{z_sub:.3f})"
            if self.chk_fit_log.isChecked():
                self.statusBar().showMessage(log)
            peak_x = ox + x_sub_v * dx_v
            peak_y = oy + y_sub * dy_v
            peak_z = oz + z_sub * dz_v
            peak_px = y_sub   # Y pixel
            peak_py = z_sub   # Z pixel

        if self.measure_type == 'distance':
            self._handle_distance_click(peak_x, peak_y, peak_z, peak_px, peak_py, plane)
        else:
            self._handle_angle_click(peak_x, peak_y, peak_z, peak_px, peak_py, plane)

    def _handle_distance_click(self, peak_x, peak_y, peak_z, peak_px, peak_py, plane='xy'):
        """2点距離モードのクリックを1回処理する。"""
        canvas = self._get_canvas(plane)
        point_num = len(self.click_points) + 1
        color = '#89b4fa' if point_num % 2 == 1 else '#f9e2af'

        # キャンバス上の表示座標 (plane によって軸の意味が違う)
        cx, cy = self._to_canvas_coords(peak_x, peak_y, peak_z, plane)
        canvas.add_click_marker(cx, cy, f"P{point_num}", color)
        self.click_points.append({
            'x': peak_x, 'y': peak_y, 'z': peak_z,
            'px': peak_px, 'py': peak_py,
            'cx': cx, 'cy': cy, 'plane': plane,
        })
        self.lbl_click_info.setText(
            f"PT{point_num}  ({peak_x:.3f}, {peak_y:.3f}, {peak_z:.3f}) Å")

        if len(self.click_points) >= 2:
            p1, p2 = self.click_points[-2], self.click_points[-1]
            # 平面内の投影距離
            dist = np.sqrt((p2['cx'] - p1['cx'])**2 + (p2['cy'] - p1['cy'])**2)
            dist_text = f"{dist:.3f} Å"
            canvas.add_distance_line(p1['cx'], p1['cy'], p2['cx'], p2['cy'], dist_text)

            row = self.table_dist.rowCount()
            self.table_dist.insertRow(row)
            self.table_dist.setItem(row, 0, QTableWidgetItem(
                f"({p1['x']:.3f}, {p1['y']:.3f}, {p1['z']:.3f})"))
            self.table_dist.setItem(row, 1, QTableWidgetItem(
                f"({p2['x']:.3f}, {p2['y']:.3f}, {p2['z']:.3f})"))
            self.table_dist.setItem(row, 2, QTableWidgetItem(dist_text))

            self.measurements.append({
                'p1_x': p1['x'], 'p1_y': p1['y'], 'p1_z': p1['z'],
                'p1_px': p1['px'], 'p1_py': p1['py'],
                'p2_x': p2['x'], 'p2_y': p2['y'], 'p2_z': p2['z'],
                'p2_px': p2['px'], 'p2_py': p2['py'],
                'distance': dist, 'plane': plane,
            })
            self.lbl_click_info.setText(
                f"DIST  {dist_text}  ·  P1 ({p1['x']:.3f}, {p1['y']:.3f}, {p1['z']:.3f})  "
                f"P2 ({p2['x']:.3f}, {p2['y']:.3f}, {p2['z']:.3f})")
            self.statusBar().showMessage(f"距離測定: {dist_text}")
            self.click_points = []

    def _to_canvas_coords(self, x, y, z, plane):
        """3D Å 座標を指定平面のキャンバス座標 (cx, cy) に変換する。"""
        if plane == 'xy':
            return x, y
        elif plane == 'xz':
            return x, z
        else:  # 'yz'
            return y, z

    def _handle_angle_click(self, peak_x, peak_y, peak_z, peak_px, peak_py, plane='xy'):
        """3点角度モードのクリックを1回処理する。"""
        canvas = self._get_canvas(plane)
        _colors = ['#89b4fa', '#ff79c6', '#f9e2af']
        _labels = ['A1', 'A2', 'A3']
        n = len(self.click_points)
        color, label = _colors[n % 3], _labels[n % 3]

        cx, cy = self._to_canvas_coords(peak_x, peak_y, peak_z, plane)
        canvas.add_click_marker(cx, cy, label, color)
        self.click_points.append({
            'x': peak_x, 'y': peak_y, 'z': peak_z,
            'px': peak_px, 'py': peak_py,
            'cx': cx, 'cy': cy, 'plane': plane,
        })
        self._update_measure_status_label()
        self.lbl_click_info.setText(
            f"{label}  ({peak_x:.3f}, {peak_y:.3f}, {peak_z:.3f}) Å")

        if len(self.click_points) >= 3:
            p1, p2, p3 = self.click_points[-3], self.click_points[-2], self.click_points[-1]
            angle_deg = self._calc_angle_3d(
                (p1['x'], p1['y'], p1['z']),
                (p2['x'], p2['y'], p2['z']),
                (p3['x'], p3['y'], p3['z']),
            )
            d12 = np.sqrt((p2['x']-p1['x'])**2 + (p2['y']-p1['y'])**2 + (p2['z']-p1['z'])**2)
            d23 = np.sqrt((p3['x']-p2['x'])**2 + (p3['y']-p2['y'])**2 + (p3['z']-p2['z'])**2)

            canvas.add_angle_lines(
                p1['cx'], p1['cy'], p2['cx'], p2['cy'], p3['cx'], p3['cy'], angle_deg)

            row = self.table_angle.rowCount()
            self.table_angle.insertRow(row)
            self.table_angle.setItem(row, 0, QTableWidgetItem(
                f"({p1['x']:.3f}, {p1['y']:.3f}, {p1['z']:.3f})"))
            self.table_angle.setItem(row, 1, QTableWidgetItem(
                f"({p2['x']:.3f}, {p2['y']:.3f}, {p2['z']:.3f})"))
            self.table_angle.setItem(row, 2, QTableWidgetItem(
                f"({p3['x']:.3f}, {p3['y']:.3f}, {p3['z']:.3f})"))
            self.table_angle.setItem(row, 3, QTableWidgetItem(f"{angle_deg:.1f}°"))
            self.table_angle.setItem(row, 4, QTableWidgetItem(f"{d12:.3f}"))
            self.table_angle.setItem(row, 5, QTableWidgetItem(f"{d23:.3f}"))

            self.angle_measurements.append({
                'p1_x': p1['x'], 'p1_y': p1['y'], 'p1_z': p1['z'],
                'p1_px': p1['px'], 'p1_py': p1['py'],
                'p2_x': p2['x'], 'p2_y': p2['y'], 'p2_z': p2['z'],
                'p2_px': p2['px'], 'p2_py': p2['py'],
                'p3_x': p3['x'], 'p3_y': p3['y'], 'p3_z': p3['z'],
                'p3_px': p3['px'], 'p3_py': p3['py'],
                'angle': angle_deg, 'd12': d12, 'd23': d23, 'plane': plane,
            })
            self.lbl_click_info.setText(
                f"∠A1-A2-A3 = {angle_deg:.1f}°  |  d12={d12:.3f} Å  d23={d23:.3f} Å")
            self.statusBar().showMessage(f"角度測定: {angle_deg:.1f}°")
            self.click_points = []
            self._update_measure_status_label()

    @staticmethod
    def _calc_angle_3d(p1, p2, p3):
        """Compute ∠P1-P2-P3 in degrees (P2 is the vertex), using 3D vectors."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_t)))

    def _redraw_measurements(self):
        """全キャンバスの測定マーカーを再描画する (全平面一括版)。"""
        for plane in ('xy', 'xz', 'yz'):
            self._redraw_measurements_on_canvas(plane)

    def _redraw_measurements_on_canvas(self, plane):
        """指定平面のキャンバスの確定済み測定マーカーを再描画する。"""
        canvas = self._get_canvas(plane)
        canvas.clear_canvas_markers()
        if self.volume is None:
            return

        m = self.metadata
        dx_v, dy_v = m['voxel_size'][2], m['voxel_size'][1]
        ox, oy = m['x_range'][0], m['y_range'][0]
        freeze = self.chk_freeze_markers.isChecked()

        # 確定済み距離測定
        for i, meas in enumerate(self.measurements):
            if meas.get('plane', 'xy') != plane:
                continue
            if plane == 'xy' and not freeze:
                # XY のみ現スライスで再探索 (freeze=OFF 時)
                data_2d = self.volume[self.slider_z.value(), :, :]
                search_r = self.spin_search_radius.value()
                p1_px, p1_py = self._find_peak(data_2d, meas['p1_px'], meas['p1_py'], search_r)
                p2_px, p2_py = self._find_peak(data_2d, meas['p2_px'], meas['p2_py'], search_r)
                p1_x = ox + p1_px * dx_v; p1_y = oy + p1_py * dy_v
                p2_x = ox + p2_px * dx_v; p2_y = oy + p2_py * dy_v
                dist = np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                c1x, c1y = p1_x, p1_y
                c2x, c2y = p2_x, p2_y
            else:
                # XZ/YZ は常に保存済みÅ座標から再描画 (freeze と同等)
                c1x, c1y = self._to_canvas_coords(
                    meas['p1_x'], meas['p1_y'], meas.get('p1_z', 0.0), plane)
                c2x, c2y = self._to_canvas_coords(
                    meas['p2_x'], meas['p2_y'], meas.get('p2_z', 0.0), plane)
                dist = meas['distance']
            canvas.add_click_marker(c1x, c1y, f"P{i*2+1}", '#89b4fa')
            canvas.add_click_marker(c2x, c2y, f"P{i*2+2}", '#f9e2af')
            canvas.add_distance_line(c1x, c1y, c2x, c2y, f"{dist:.3f} Å")

        # 確定済み角度測定
        for ameas in self.angle_measurements:
            if ameas.get('plane', 'xy') != plane:
                continue
            if plane == 'xy' and not freeze:
                data_2d = self.volume[self.slider_z.value(), :, :]
                search_r = self.spin_search_radius.value()
                p1_px, p1_py = self._find_peak(data_2d, ameas['p1_px'], ameas['p1_py'], search_r)
                p2_px, p2_py = self._find_peak(data_2d, ameas['p2_px'], ameas['p2_py'], search_r)
                p3_px, p3_py = self._find_peak(data_2d, ameas['p3_px'], ameas['p3_py'], search_r)
                p1_x = ox + p1_px * dx_v; p1_y = oy + p1_py * dy_v
                p2_x = ox + p2_px * dx_v; p2_y = oy + p2_py * dy_v
                p3_x = ox + p3_px * dx_v; p3_y = oy + p3_py * dy_v
                angle_deg = self._calc_angle_3d(
                    (p1_x, p1_y, 0.0), (p2_x, p2_y, 0.0), (p3_x, p3_y, 0.0))
                c1x, c1y = p1_x, p1_y
                c2x, c2y = p2_x, p2_y
                c3x, c3y = p3_x, p3_y
            else:
                c1x, c1y = self._to_canvas_coords(
                    ameas['p1_x'], ameas['p1_y'], ameas['p1_z'], plane)
                c2x, c2y = self._to_canvas_coords(
                    ameas['p2_x'], ameas['p2_y'], ameas['p2_z'], plane)
                c3x, c3y = self._to_canvas_coords(
                    ameas['p3_x'], ameas['p3_y'], ameas['p3_z'], plane)
                angle_deg = ameas['angle']
            canvas.add_angle_lines(c1x, c1y, c2x, c2y, c3x, c3y, angle_deg)

        # 進行中クリック点 (plane が一致するもののみ)
        _dist_colors = ['#89b4fa', '#f9e2af']
        _angle_colors = ['#89b4fa', '#ff79c6', '#f9e2af']
        _angle_labels = ['A1', 'A2', 'A3']
        for i, pt in enumerate(self.click_points):
            if pt.get('plane') != plane:
                continue
            if plane == 'xy' and not freeze:
                data_2d = self.volume[self.slider_z.value(), :, :]
                search_r = self.spin_search_radius.value()
                new_px, new_py = self._find_peak(data_2d, pt['px'], pt['py'], search_r)
                new_cx = ox + new_px * dx_v
                new_cy = oy + new_py * dy_v
            else:
                new_cx, new_cy = pt.get('cx', pt['x']), pt.get('cy', pt['y'])
            if self.measure_type == 'distance':
                color = _dist_colors[i % 2]
                canvas.add_click_marker(
                    new_cx, new_cy, f"P{len(self.measurements)*2+i+1}", color)
            else:
                canvas.add_click_marker(
                    new_cx, new_cy, _angle_labels[i % 3], _angle_colors[i % 3])

        canvas.draw_idle()

    def _find_peak(self, data_2d, px, py, search_radius):
        """クリック周辺のピークをサブピクセル精度で検出"""
        ny, nx = data_2d.shape
        ipx, ipy = int(round(px)), int(round(py))

        x_min = max(0, ipx - search_radius)
        x_max = min(nx, ipx + search_radius + 1)
        y_min = max(0, ipy - search_radius)
        y_max = min(ny, ipy + search_radius + 1)

        region = data_2d[y_min:y_max, x_min:x_max]
        if region.size == 0:
            return float(ipx), float(ipy)

        local_y, local_x = np.unravel_index(np.argmax(region), region.shape)
        peak_ix = x_min + local_x
        peak_iy = y_min + local_y

        # サブピクセル: 重心法（3x3近傍）
        sub_r = 1
        sx_min = max(0, peak_ix - sub_r)
        sx_max = min(nx, peak_ix + sub_r + 1)
        sy_min = max(0, peak_iy - sub_r)
        sy_max = min(ny, peak_iy + sub_r + 1)

        sub = data_2d[sy_min:sy_max, sx_min:sx_max].astype(float)
        sub = sub - sub.min()
        total = sub.sum()

        if total > 0:
            yy, xx = np.mgrid[sy_min:sy_max, sx_min:sx_max]
            return (xx * sub).sum() / total, (yy * sub).sum() / total
        return float(peak_ix), float(peak_iy)

    def _on_reset_dist(self):
        """距離タブの選択リセット: 全平面の進行中選択 + 履歴 + マーカーを全クリア。"""
        self.click_points = []
        self.measurements = []
        self.table_dist.setRowCount(0)
        for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
            cv.clear_measurements()
        self._redraw_measurements()
        self.lbl_click_info.setText("")
        self.statusBar().showMessage("距離: 選択をリセットしました")

    def _on_reset_angle(self):
        """角度タブの選択リセット: 全平面の進行中選択 + 履歴 + マーカーを全クリア。"""
        self.click_points = []
        self.angle_measurements = []
        self.table_angle.setRowCount(0)
        for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
            cv.clear_measurements()
        self._redraw_measurements()
        self.lbl_click_info.setText("")
        self.statusBar().showMessage("角度: 選択をリセットしました")

    def _clear_current_tab(self):
        """現在アクティブな結果タブの測定をクリアする (3平面すべてに作用)。"""
        tab_idx = self.tab_results.currentIndex()
        if tab_idx == 0:
            self.click_points = []
            self.measurements = []
            self.table_dist.setRowCount(0)
            for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
                cv.clear_measurements()
            self._redraw_measurements()
            self.lbl_click_info.setText("")
            self.statusBar().showMessage("距離測定履歴をクリアしました")
        elif tab_idx == 1:
            self.click_points = []
            self.angle_measurements = []
            self.table_angle.setRowCount(0)
            for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
                cv.clear_measurements()
            self._redraw_measurements()
            self.lbl_click_info.setText("")
            self.statusBar().showMessage("角度測定履歴をクリアしました")
        else:
            self.detected_peaks = []
            self.table_peaks.setRowCount(0)
            for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
                cv.clear_peak_markers()
                cv.draw_idle()
            self.statusBar().showMessage("検出ピークをクリアしました")

    def _export_distance_csv(self):
        if not self.measurements:
            QMessageBox.information(self, "情報", "エクスポートする距離測定結果がありません")
            return
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath, _ = QFileDialog.getSaveFileName(
            self, "距離CSVエクスポート", f"distances_{ts}.csv", "CSV files (*.csv)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8-sig') as f:
                f.write("P1_X(Å),P1_Y(Å),P1_Z(Å),P2_X(Å),P2_Y(Å),P2_Z(Å),Distance(Å),Plane\n")
                for m in self.measurements:
                    f.write(f"{m['p1_x']:.4f},{m['p1_y']:.4f},{m.get('p1_z', 0.0):.4f},"
                            f"{m['p2_x']:.4f},{m['p2_y']:.4f},{m.get('p2_z', 0.0):.4f},"
                            f"{m['distance']:.4f},{m['plane']}\n")
            self.statusBar().showMessage(f"距離CSVエクスポート完了: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"エクスポート失敗:\n{str(e)}")

    def _export_angle_csv(self):
        if not self.angle_measurements:
            QMessageBox.information(self, "情報", "エクスポートする角度測定結果がありません")
            return
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath, _ = QFileDialog.getSaveFileName(
            self, "角度CSVエクスポート", f"angles_{ts}.csv", "CSV files (*.csv)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8-sig') as f:
                f.write("P1_X(Å),P1_Y(Å),P1_Z(Å),P2_X(Å),P2_Y(Å),P2_Z(Å),"
                        "P3_X(Å),P3_Y(Å),P3_Z(Å),Angle(deg),d12(Å),d23(Å),Plane\n")
                for a in self.angle_measurements:
                    f.write(
                        f"{a['p1_x']:.4f},{a['p1_y']:.4f},{a['p1_z']:.4f},"
                        f"{a['p2_x']:.4f},{a['p2_y']:.4f},{a['p2_z']:.4f},"
                        f"{a['p3_x']:.4f},{a['p3_y']:.4f},{a['p3_z']:.4f},"
                        f"{a['angle']:.1f},{a['d12']:.4f},{a['d23']:.4f},{a['plane']}\n"
                    )
            self.statusBar().showMessage(f"角度CSVエクスポート完了: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"エクスポート失敗:\n{str(e)}")

    def _export_peaks_csv(self):
        if not self.detected_peaks:
            QMessageBox.information(self, "情報", "エクスポートする検出ピークがありません")
            return
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath, _ = QFileDialog.getSaveFileName(
            self, "ピークCSVエクスポート", f"peaks_{ts}.csv", "CSV files (*.csv)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8-sig') as f:
                f.write("No.,X(Å),Y(Å),Z(Å),Intensity\n")
                for i, pk in enumerate(self.detected_peaks):
                    f.write(f"{i+1},{pk['x']:.4f},{pk['y']:.4f},{pk['z']:.4f},"
                            f"{pk['intensity']:.6f}\n")
            self.statusBar().showMessage(f"ピークCSVエクスポート完了: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"エクスポート失敗:\n{str(e)}")


    # -------------------------------------------------------------------------
    # ピーク自動検出
    # -------------------------------------------------------------------------
    def _on_peak_range_changed(self, _checked):
        """Enable/disable Z range spinboxes based on the selected range radio."""
        enabled = self.radio_peak_zrange.isChecked()
        self.spin_peak_z_start.setEnabled(enabled)
        self.spin_peak_z_end.setEnabled(enabled)

    def _on_detect_peaks(self):
        if self.volume is None:
            QMessageBox.information(self, "情報", "ボリュームデータを読み込んでから実行してください")
            return

        m = self.metadata
        dz_v, dy_v, dx_v = m['voxel_size']
        oz = m['z_range'][0]
        percentile = self.spin_peak_percentile.value()
        min_dist_ang = self.spin_peak_min_dist.value()
        range_id = self.btn_grp_peak_range.checkedId()

        if range_id == 0:
            z_idx = self.slider_z.value()
            volume_region = self.volume[z_idx:z_idx+1, :, :]
            z_offset = z_idx
        elif range_id == 1:
            z_start = self.spin_peak_z_start.value()
            z_end = self.spin_peak_z_end.value()
            zi_start = max(0, int((z_start - oz) / dz_v))
            zi_end = min(self.volume.shape[0], int((z_end - oz) / dz_v) + 1)
            if zi_start >= zi_end:
                QMessageBox.warning(self, "警告", "Z範囲が不正です (開始 ≥ 終了)")
                return
            volume_region = self.volume[zi_start:zi_end, :, :]
            z_offset = zi_start
        else:
            volume_region = self.volume
            z_offset = 0

        if volume_region.size > 50_000_000:
            self._start_peak_detection_thread(volume_region, z_offset, percentile, min_dist_ang)
        else:
            self._run_peak_detection_sync(volume_region, z_offset, percentile, min_dist_ang)

    def _run_peak_detection_sync(self, volume_region, z_offset, percentile, min_dist_ang):
        m = self.metadata
        dz_v, dy_v, dx_v = m['voxel_size']
        oz, oy, ox = m['z_range'][0], m['y_range'][0], m['x_range'][0]
        filter_size = (
            max(1, 2 * max(1, int(round(min_dist_ang / dz_v))) + 1),
            max(1, 2 * max(1, int(round(min_dist_ang / dy_v))) + 1),
            max(1, 2 * max(1, int(round(min_dist_ang / dx_v))) + 1),
        )
        vol = volume_region.astype(np.float32)
        filtered = maximum_filter(vol, size=filter_size)
        threshold = np.percentile(vol, percentile)
        local_max = (vol == filtered) & (vol > threshold)
        zz, yy, xx = np.where(local_max)

        refine_mode = self.combo_peak_refine.currentIndex()
        fit_size = self.spin_fit_size.value()
        if len(zz) >= 100 and refine_mode != PEAK_REFINE_OFF:
            self.statusBar().showMessage(f"ピーク精緻化中 ({len(zz)} 個)…")
            QApplication.processEvents()

        peaks = []
        for zi, yi, xi in zip(zz, yy, xx):
            abs_zi = int(zi) + z_offset
            if refine_mode != PEAK_REFINE_OFF:
                x_sub, y_sub, z_sub = _apply_peak_refinement_on_region(
                    vol, int(zi), int(yi), int(xi), refine_mode, fit_size)
                abs_z_sub = z_sub + z_offset
            else:
                x_sub, y_sub, abs_z_sub = float(xi), float(yi), float(abs_zi)
            peaks.append({
                'x': ox + x_sub * dx_v,
                'y': oy + y_sub * dy_v,
                'z': oz + abs_z_sub * dz_v,
                'px': x_sub, 'py': y_sub, 'pz': abs_z_sub,
                'intensity': float(vol[int(zi), int(yi), int(xi)]),
            })
        self.detected_peaks = peaks
        self._populate_peaks_table()
        self._update_peak_display()
        self.statusBar().showMessage(f"ピーク検出完了: {len(peaks)} 個")

    def _start_peak_detection_thread(self, volume_region, z_offset, percentile, min_dist_ang):
        if self._peak_worker is not None and self._peak_worker.isRunning():
            return
        self.btn_detect_peaks.setEnabled(False)
        self.statusBar().showMessage("ピーク検出中 (バックグラウンド)...")
        self._peak_worker = PeakDetectionWorker(
            volume_region, z_offset, self.metadata, percentile, min_dist_ang,
            refine_mode=self.combo_peak_refine.currentIndex(),
            fit_size=self.spin_fit_size.value())
        self._peak_worker.finished.connect(self._on_peak_detection_done)
        self._peak_worker.start()

    def _on_peak_detection_done(self, peaks):
        self.detected_peaks = peaks
        self.btn_detect_peaks.setEnabled(True)
        self._populate_peaks_table()
        self._update_peak_display()
        self.statusBar().showMessage(f"ピーク検出完了: {len(peaks)} 個")
        self._peak_worker = None

    def _populate_peaks_table(self):
        self.table_peaks.setRowCount(0)
        for i, pk in enumerate(self.detected_peaks):
            row = self.table_peaks.rowCount()
            self.table_peaks.insertRow(row)
            self.table_peaks.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self.table_peaks.setItem(row, 1, QTableWidgetItem(f"{pk['x']:.3f}"))
            self.table_peaks.setItem(row, 2, QTableWidgetItem(f"{pk['y']:.3f}"))
            self.table_peaks.setItem(row, 3, QTableWidgetItem(f"{pk['z']:.3f}"))
            self.table_peaks.setItem(row, 4, QTableWidgetItem(f"{pk['intensity']:.4f}"))

    def _update_peak_display(self):
        """3平面すべてのピークマーカーを更新する。"""
        for plane in ('xy', 'xz', 'yz'):
            self._update_peak_display_on_canvas(plane)

    def _update_peak_display_on_canvas(self, plane):
        """指定平面のキャンバスにピークマーカーを描画する。"""
        canvas = self._get_canvas(plane)
        if not self.chk_show_peak_markers.isChecked() or not self.detected_peaks:
            canvas.clear_peak_markers()
            if self.detected_peaks:
                canvas.draw_idle()
            return
        # 各平面の2D投影座標でマーカーを表示
        if plane == 'xy':
            peaks_2d = [(pk['x'], pk['y']) for pk in self.detected_peaks]
        elif plane == 'xz':
            peaks_2d = [(pk['x'], pk['z']) for pk in self.detected_peaks]
        else:
            peaks_2d = [(pk['y'], pk['z']) for pk in self.detected_peaks]
        canvas.draw_peak_markers(peaks_2d)

    def _on_show_peaks_changed(self, state):
        if state == Qt.Checked:
            self._update_peak_display()
        else:
            for cv in (self.canvas_xy, self.canvas_xz, self.canvas_yz):
                cv.clear_peak_markers()
                cv.draw_idle()


# =============================================================================
# デモデータ
# =============================================================================
def generate_demo_data():
    print("デモデータ生成中...")
    nx = ny = nz = 80
    dx = dy = dz = 0.1
    cx0 = nx * dx / 2  # 中心原点オフセット = 4.0 Å
    # ガウスブロブの絶対位置 (ピクセル計算用、0〜8 Å 空間)
    abs_positions = [
        (4.0, 4.0, 4.0), (4.0, 2.0, 2.0), (2.0, 4.0, 2.0), (2.0, 2.0, 4.0),
        (0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 4.0),
        (6.0, 4.0, 2.0), (6.0, 2.0, 4.0), (2.0, 6.0, 4.0), (4.0, 6.0, 2.0),
    ]
    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    sigma = 3.0
    for ax, ay, az in abs_positions:
        pcx, pcy, pcz = ax/dx, ay/dy, az/dz
        for zi in range(max(0, int(pcz)-10), min(nz, int(pcz)+11)):
            for yi in range(max(0, int(pcy)-10), min(ny, int(pcy)+11)):
                for xi in range(max(0, int(pcx)-10), min(nx, int(pcx)+11)):
                    r2 = (xi-pcx)**2 + (yi-pcy)**2 + (zi-pcz)**2
                    volume[zi, yi, xi] += np.exp(-r2/(2*sigma**2))
    volume += np.random.normal(0, 0.01, volume.shape).astype(np.float32)
    volume = np.clip(volume, 0, None)
    # 原子座標を中心原点に変換 (絶対値 - 4.0 Å)
    atoms = [{'element': 'Si', 'x': ax - cx0, 'y': ay - cx0, 'z': az - cx0}
             for ax, ay, az in abs_positions]
    return volume, atoms


# =============================================================================
# main
# =============================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText, QColor(205, 214, 244))
    palette.setColor(QPalette.Base, QColor(49, 50, 68))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 46))
    palette.setColor(QPalette.ToolTipBase, QColor(205, 214, 244))
    palette.setColor(QPalette.ToolTipText, QColor(205, 214, 244))
    palette.setColor(QPalette.Text, QColor(205, 214, 244))
    palette.setColor(QPalette.Button, QColor(49, 50, 68))
    palette.setColor(QPalette.ButtonText, QColor(205, 214, 244))
    palette.setColor(QPalette.Highlight, QColor(137, 180, 250))
    palette.setColor(QPalette.HighlightedText, QColor(30, 30, 46))
    app.setPalette(palette)

    window = AtomViewerWindow()

    if '--demo' in sys.argv:
        volume, atoms = generate_demo_data()
        window._set_volume_data(volume, "デモデータ (FCC Si)")
        elements = set(a['element'] for a in atoms)
        demo_set = {
            'filepath': None,
            'name': 'デモ (FCC Si)',
            'atoms': atoms,
            'visible': True,
            'offset': [0.0, 0.0, 0.0],
            'element_colors': {e: get_element_color(e) for e in elements},
            'element_radii': {e: get_element_radius(e) for e in elements},
        }
        window.atom_sets.append(demo_set)
        window.active_atom_set_index = 0
        window._rebuild_xyz_list_ui()
        window._update_active_ui()
        window._update_xyz_info_label()
        window._update_atom_overlay()
        window.statusBar().showMessage("デモモードで起動")

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()