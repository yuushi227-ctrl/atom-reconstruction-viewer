
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
        QListWidgetItem, QScrollArea, QColorDialog
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QThread, pyqtSlot
    from PyQt5.QtGui import QFont, QColor, QIcon
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
        for obj in self.click_markers:
            try:
                obj.remove()
            except Exception:
                pass
        self.click_markers.clear()
        for obj in self.distance_lines:
            try:
                obj.remove()
            except Exception:
                pass
        self.distance_lines.clear()
        for obj in self.distance_texts:
            try:
                obj.remove()
            except Exception:
                pass
        self.distance_texts.clear()

    def clear_measurements(self):
        self.clear_canvas_markers()
        self.draw_idle()


# =============================================================================
# メインウィンドウ
# =============================================================================
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
        self.atoms = []
        self.click_points = []
        self.measurements = []
        self.element_settings = {}   # {element: {'color': '#hex', 'radius': float}}

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

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("border-top: 1px solid #061828;")
        file_layout.addWidget(sep)

        self.btn_load_xyz = QPushButton("XYZファイル読み込み")
        file_layout.addWidget(self.btn_load_xyz)

        self.lbl_xyz_info = QLabel("— 未読み込み —")
        self.lbl_xyz_info.setStyleSheet("color: #90a8c0; font-size: 11px;")
        file_layout.addWidget(self.lbl_xyz_info)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

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
        slice_group = QGroupBox("▸ SLICE  Z-AXIS")
        slice_layout = QGridLayout()

        slice_layout.addWidget(QLabel("XY平面 (Z):"), 0, 0)
        self.slider_z = QSlider(Qt.Horizontal)
        self.spin_z = QSpinBox()
        slice_layout.addWidget(self.slider_z, 0, 1)
        slice_layout.addWidget(self.spin_z, 0, 2)

        slice_layout.addWidget(QLabel("Z原点 (slice):"), 1, 0)
        self.spin_z_origin = QSpinBox()
        self.spin_z_origin.setRange(0, 9999)
        self.spin_z_origin.setValue(0)
        self.spin_z_origin.setToolTip("この番号のスライスをZ=0・slice 0として表示します")
        slice_layout.addWidget(self.spin_z_origin, 1, 1, 1, 2)

        slice_group.setLayout(slice_layout)
        left_layout.addWidget(slice_group)

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

        thick_row = QHBoxLayout()
        thick_row.addWidget(QLabel("原子表示厚み (Å):"))
        self.spin_atom_thickness = QDoubleSpinBox()
        self.spin_atom_thickness.setRange(0.1, 50.0)
        self.spin_atom_thickness.setValue(2.0)
        self.spin_atom_thickness.setSingleStep(0.5)
        thick_row.addWidget(self.spin_atom_thickness)
        display_layout.addLayout(thick_row)

        # クラスターZ位置 (ボリュームZとは独立)
        cluster_z_row = QHBoxLayout()
        cluster_z_row.addWidget(QLabel("クラスターZ (Å):"))
        self.spin_cluster_z = QDoubleSpinBox()
        self.spin_cluster_z.setRange(-1000.0, 1000.0)
        self.spin_cluster_z.setDecimals(3)
        self.spin_cluster_z.setValue(0.0)
        self.spin_cluster_z.setSingleStep(0.1)
        cluster_z_row.addWidget(self.spin_cluster_z)
        display_layout.addLayout(cluster_z_row)

        self.chk_link_cluster_z = QCheckBox("ボリュームZと連動")
        self.chk_link_cluster_z.setChecked(False)
        display_layout.addWidget(self.chk_link_cluster_z)

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

        # --- 距離測定 ---
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

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("ピーク探索半径 (px):"))
        self.spin_search_radius = QSpinBox()
        self.spin_search_radius.setRange(1, 50)
        self.spin_search_radius.setValue(15)
        search_row.addWidget(self.spin_search_radius)
        measure_layout.addLayout(search_row)

        self.btn_clear_measure = QPushButton("✕  CLEAR")
        self.btn_clear_measure.setObjectName("danger")
        measure_layout.addWidget(self.btn_clear_measure)

        measure_group.setLayout(measure_layout)
        left_layout.addWidget(measure_group)

        # --- 測定結果テーブル ---
        result_group = QGroupBox("▸ RESULTS")
        result_layout = QVBoxLayout()

        self.table_results = QTableWidget(0, 3)
        self.table_results.setHorizontalHeaderLabels(["点1 (Å)", "点2 (Å)", "距離 (Å)"])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_results.setMaximumHeight(160)
        result_layout.addWidget(self.table_results)

        self.btn_export_csv = QPushButton("↓  CSV EXPORT")
        result_layout.addWidget(self.btn_export_csv)

        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)

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
        self.canvas_xy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas_xy, stretch=1)

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
        self.btn_load_xyz.clicked.connect(self._on_load_xyz)
        self.btn_apply_voxel.clicked.connect(self._on_apply_voxel)

        self.slider_z.valueChanged.connect(self.spin_z.setValue)
        self.spin_z.valueChanged.connect(self.slider_z.setValue)
        self.spin_z.valueChanged.connect(lambda: self._update_slice())
        self.spin_z_origin.valueChanged.connect(lambda: self._update_slice())

        self.combo_cmap.currentTextChanged.connect(self._update_all_slices)
        self.chk_auto_intensity.stateChanged.connect(self._on_auto_intensity_changed)
        self.spin_pct_high.valueChanged.connect(self._on_pct_high_changed)
        self.spin_pct_low.valueChanged.connect(self._on_pct_low_changed)
        self.spin_vmin.valueChanged.connect(self._update_all_slices)
        self.spin_vmax.valueChanged.connect(self._update_all_slices)
        self.btn_intensity_from_slice.clicked.connect(self._on_intensity_from_slice)
        self.btn_intensity_from_volume.clicked.connect(self._on_intensity_from_volume)
        self.chk_show_atoms.stateChanged.connect(self._update_atom_overlay)
        self.spin_atom_thickness.valueChanged.connect(self._update_atom_overlay)
        self.spin_cluster_z.valueChanged.connect(self._update_atom_overlay)
        self.chk_link_cluster_z.stateChanged.connect(self._on_link_cluster_z_changed)

        self.btn_toggle_measure.toggled.connect(self._toggle_measure_mode)
        self.btn_clear_measure.clicked.connect(self._clear_measurements)
        self.btn_export_csv.clicked.connect(self._export_csv)

        self.canvas_xy.point_clicked.connect(lambda x, y: self._on_canvas_click(x, y))

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

        # XY は中心原点（クラスターXYZ の座標系に合わせる）
        # Z は 0 始まり（スライス番号との対応を保つ）
        self.metadata = {
            'voxel_size': (dz, dy, dx),
            'origin': (0.0, 0.0, 0.0),
            'x_range': (-nx * dx / 2, nx * dx / 2),
            'y_range': (-ny * dy / 2, ny * dy / 2),
            'z_range': (0.0, nz * dz),
        }

        self.slider_z.setRange(0, nz - 1)
        self.spin_z.setRange(0, nz - 1)
        self.slider_z.setValue(nz // 2)
        self.spin_z_origin.setRange(0, nz - 1)
        self.spin_z_origin.setValue(80 if nz > 80 else 0)

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
        self.metadata['z_range'] = (0.0, nz * dz)
        self._update_all_slices()
        self.statusBar().showMessage(f"ボクセルサイズ更新: dx={dx}, dy={dy}, dz={dz} Å")

    # -------------------------------------------------------------------------
    # XYZ読み込み
    # -------------------------------------------------------------------------
    def _on_load_xyz(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "XYZファイルを選択", "", "XYZ files (*.xyz);;All files (*)")
        if not filepath:
            return
        try:
            self.atoms = load_xyz(filepath)
            if not self.atoms:
                raise ValueError("原子データが見つかりませんでした（フォーマットを確認してください）")
            z_vals = [a['z'] for a in self.atoms]
            z_min, z_max = min(z_vals), max(z_vals)
            median_z = float(np.median(z_vals))
            # クラスターZをロードした原子のZ中央値に自動設定
            self.spin_cluster_z.blockSignals(True)
            self.spin_cluster_z.setValue(median_z)
            self.spin_cluster_z.blockSignals(False)
            self.lbl_xyz_info.setText(
                f"{len(self.atoms)}個の原子 | {Path(filepath).name}\n"
                f"Z範囲: {z_min:.2f} ～ {z_max:.2f} Å (中央: {median_z:.2f})"
            )
            self._rebuild_element_settings_ui()
            self._update_atom_overlay()
            self.statusBar().showMessage(f"XYZ読み込み完了: {len(self.atoms)}個の原子")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"XYZ読み込み失敗:\n{str(e)}")

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
        if self.volume is None:
            return
        cmap = self.combo_cmap.currentText()
        idx = self.slider_z.value()
        data = self.volume[idx, :, :]
        extent = self._get_extent('xy')
        dz = self.metadata['voxel_size'][0]
        z_origin = self.spin_z_origin.value()
        rel_idx = idx - z_origin          # 表示スライス番号（負も可）
        z_pos = rel_idx * dz              # 表示Z座標（負も可）
        self.canvas_xy.ax.set_xlabel("X (Å)", color='#4a6880', fontsize=11)
        self.canvas_xy.ax.set_ylabel("Y (Å)", color='#4a6880', fontsize=11)
        self.canvas_xy.ax.set_title(
            f"XY  ·  Z = {z_pos:+.3f} Å  ·  slice {rel_idx:+d}",
            color='#0077b6', fontsize=13, fontweight='bold')
        vmin, vmax = self._compute_vmin_vmax(data)
        self.canvas_xy.display_slice(data, extent, cmap, vmin=vmin, vmax=vmax)
        # ボリュームZと連動している場合はクラスターZも更新
        if self.chk_link_cluster_z.isChecked():
            self.spin_cluster_z.blockSignals(True)
            self.spin_cluster_z.setValue(z_pos)
            self.spin_cluster_z.blockSignals(False)
        self._update_atom_overlay_for_plane()
        self._redraw_measurements()

    def _update_all_slices(self):
        self._update_slice()

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
        data = self.volume[self.slider_z.value(), :, :]
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

    # 原子オーバーレイ
    # -------------------------------------------------------------------------
    def _update_atom_overlay(self):
        self._update_atom_overlay_for_plane()

    def _update_atom_overlay_for_plane(self):
        if self.volume is None:
            return
        self.canvas_xy.clear_atoms()

        if not self.atoms or not self.chk_show_atoms.isChecked():
            return

        thickness = self.spin_atom_thickness.value()
        cluster_z = self.spin_cluster_z.value()
        positions, colors, radii = [], [], []

        for a in self.atoms:
            if abs(a['z'] - cluster_z) <= thickness / 2:
                elem = a['element']
                s = self.element_settings.get(elem, {})
                positions.append((a['x'], a['y']))
                colors.append(s.get('color', get_element_color(elem)))
                radii.append(s.get('radius', get_element_radius(elem)))

        if positions:
            self.canvas_xy.overlay_atoms(positions, colors, radii)

    def _rebuild_element_settings_ui(self):
        """XYZ読み込み後、検出元素ごとの色・半径設定行を再構築する"""
        # 既存行を削除
        while self.elem_rows_layout.count():
            item = self.elem_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        elements = sorted(set(a['element'] for a in self.atoms))
        if not elements:
            self.elem_rows_layout.addWidget(QLabel("XYZファイルを読み込んでください"))
            return

        for elem in elements:
            if elem not in self.element_settings:
                self.element_settings[elem] = {
                    'color': get_element_color(elem),
                    'radius': get_element_radius(elem),
                }
            s = self.element_settings[elem]

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 1, 0, 1)
            row_layout.setSpacing(4)

            # 元素名ラベル
            lbl = QLabel(elem)
            lbl.setFixedWidth(42)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"font-weight: bold; color: {s['color']}; font-size: 12px;"
            )
            row_layout.addWidget(lbl)

            # 色ボタン
            btn_color = QPushButton()
            btn_color.setFixedSize(52, 26)
            btn_color.setStyleSheet(
                f"background-color: {s['color']}; border: 1px solid #45475a; border-radius: 3px;"
            )

            def _make_color_cb(e, btn, lbl_ref):
                def cb():
                    from PyQt5.QtGui import QColor as _QColor
                    cur = _QColor(self.element_settings[e]['color'])
                    c = QColorDialog.getColor(cur, self, f"{e} の色")
                    if c.isValid():
                        hex_c = c.name()
                        self.element_settings[e]['color'] = hex_c
                        btn.setStyleSheet(
                            f"background-color: {hex_c}; border: 1px solid #45475a; border-radius: 3px;"
                        )
                        lbl_ref.setStyleSheet(
                            f"font-weight: bold; color: {hex_c}; font-size: 13px;"
                        )
                        self._update_atom_overlay()
                return cb

            btn_color.clicked.connect(_make_color_cb(elem, btn_color, lbl))
            row_layout.addWidget(btn_color)

            # 半径スピンボックス
            spin_r = QDoubleSpinBox()
            spin_r.setRange(0.01, 20.0)
            spin_r.setDecimals(2)
            spin_r.setValue(s['radius'])
            spin_r.setSingleStep(0.1)
            spin_r.setFixedWidth(76)

            def _make_radius_cb(e):
                def cb(val):
                    self.element_settings[e]['radius'] = val
                    self._update_atom_overlay()
                return cb

            spin_r.valueChanged.connect(_make_radius_cb(elem))
            row_layout.addWidget(spin_r)
            row_layout.addStretch()

            self.elem_rows_layout.addWidget(row)

        self.elem_rows_layout.addStretch()

    def _on_link_cluster_z_changed(self, state):
        if state:
            m = self.metadata
            if m:
                idx = self.slider_z.value()
                z_pos = m['z_range'][0] + idx * m['voxel_size'][0]
                self.spin_cluster_z.setValue(z_pos)
        self._update_atom_overlay_for_plane()

    # -------------------------------------------------------------------------
    # 距離測定
    # -------------------------------------------------------------------------
    def _toggle_measure_mode(self, checked):
        self.measure_mode = checked
        if checked:
            self.lbl_measure_mode.setText("◉ MEASURE  ON  — 2点をクリック")
            self.lbl_measure_mode.setStyleSheet(
                "font-weight: bold; font-size: 11px; letter-spacing: 1px; color: #00875a;")
            self.click_points = []
        else:
            self.lbl_measure_mode.setText("● MEASURE  OFF")
            self.lbl_measure_mode.setStyleSheet(
                "font-weight: bold; font-size: 11px; letter-spacing: 2px; color: #90a8c0;")

    def _on_canvas_click(self, x_real, y_real):
        if not self.measure_mode or self.volume is None:
            return

        m = self.metadata
        dx_v, dy_v = m['voxel_size'][2], m['voxel_size'][1]
        ox, oy = m['x_range'][0], m['y_range'][0]
        data_2d = self.volume[self.slider_z.value(), :, :]

        px = (x_real - ox) / dx_v
        py = (y_real - oy) / dy_v

        search_r = self.spin_search_radius.value()
        peak_px, peak_py = self._find_peak(data_2d, px, py, search_r)

        peak_x = ox + peak_px * dx_v
        peak_y = oy + peak_py * dy_v

        point_num = len(self.click_points) + 1
        color = '#89b4fa' if point_num % 2 == 1 else '#f9e2af'
        self.canvas_xy.add_click_marker(peak_x, peak_y, f"P{point_num}", color)

        self.click_points.append({
            'x': peak_x, 'y': peak_y,
            'px': peak_px, 'py': peak_py,
        })

        self.lbl_click_info.setText(
            f"PT{point_num}  ({peak_x:.3f}, {peak_y:.3f}) Å  ·  px ({peak_px:.1f}, {peak_py:.1f})")

        if len(self.click_points) >= 2:
            p1, p2 = self.click_points[-2], self.click_points[-1]
            dist = np.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
            dist_text = f"{dist:.3f} Å"
            self.canvas_xy.add_distance_line(p1['x'], p1['y'], p2['x'], p2['y'], dist_text)

            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            self.table_results.setItem(row, 0,
                QTableWidgetItem(f"({p1['x']:.3f}, {p1['y']:.3f})"))
            self.table_results.setItem(row, 1,
                QTableWidgetItem(f"({p2['x']:.3f}, {p2['y']:.3f})"))
            self.table_results.setItem(row, 2, QTableWidgetItem(dist_text))

            self.measurements.append({
                'p1_x': p1['x'], 'p1_y': p1['y'],
                'p1_px': p1['px'], 'p1_py': p1['py'],
                'p2_x': p2['x'], 'p2_y': p2['y'],
                'p2_px': p2['px'], 'p2_py': p2['py'],
                'distance': dist, 'plane': 'xy',
            })

            self.lbl_click_info.setText(
                f"DIST  {dist_text}  ·  P1 ({p1['x']:.3f}, {p1['y']:.3f})  P2 ({p2['x']:.3f}, {p2['y']:.3f})")
            self.statusBar().showMessage(f"距離測定: {dist_text}")
            self.click_points = []

    def _redraw_measurements(self):
        """スライス変更のたびに現在Zで再ピーク検出してマーカーを更新"""
        self.canvas_xy.clear_canvas_markers()
        if self.volume is None:
            return

        m = self.metadata
        dx_v = m['voxel_size'][2]
        dy_v = m['voxel_size'][1]
        ox = m['x_range'][0]
        oy = m['y_range'][0]
        data_2d = self.volume[self.slider_z.value(), :, :]
        search_r = self.spin_search_radius.value()

        colors_p1 = '#89b4fa'
        colors_p2 = '#f9e2af'

        # 確定済み測定: 保存ピクセル座標を現在Zで再ピーク検出して再描画
        for i, meas in enumerate(self.measurements):
            p1_px, p1_py = self._find_peak(data_2d, meas['p1_px'], meas['p1_py'], search_r)
            p2_px, p2_py = self._find_peak(data_2d, meas['p2_px'], meas['p2_py'], search_r)
            p1_x = ox + p1_px * dx_v
            p1_y = oy + p1_py * dy_v
            p2_x = ox + p2_px * dx_v
            p2_y = oy + p2_py * dy_v
            dist = np.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
            self.canvas_xy.add_click_marker(p1_x, p1_y, f"P{i*2+1}", colors_p1)
            self.canvas_xy.add_click_marker(p2_x, p2_y, f"P{i*2+2}", colors_p2)
            self.canvas_xy.add_distance_line(p1_x, p1_y, p2_x, p2_y, f"{dist:.3f} Å")

        # 未確定の1点目も現在Zで再ピーク検出
        for i, pt in enumerate(self.click_points):
            new_px, new_py = self._find_peak(data_2d, pt['px'], pt['py'], search_r)
            new_x = ox + new_px * dx_v
            new_y = oy + new_py * dy_v
            color = '#89b4fa' if i % 2 == 0 else '#f9e2af'
            self.canvas_xy.add_click_marker(
                new_x, new_y, f"P{len(self.measurements)*2+i+1}", color
            )

        self.canvas_xy.draw_idle()

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

    def _clear_measurements(self):
        self.click_points = []
        self.measurements = []
        self.table_results.setRowCount(0)
        self.canvas_xy.clear_measurements()
        self.lbl_click_info.setText("")
        self.statusBar().showMessage("測定結果をクリアしました")

    def _export_csv(self):
        if not self.measurements:
            QMessageBox.information(self, "情報", "エクスポートする測定結果がありません")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "CSVエクスポート", "measurements.csv", "CSV files (*.csv)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8-sig') as f:
                f.write("P1_X(Å),P1_Y(Å),P2_X(Å),P2_Y(Å),Distance(Å),Plane\n")
                for m in self.measurements:
                    f.write(f"{m['p1_x']:.4f},{m['p1_y']:.4f},"
                            f"{m['p2_x']:.4f},{m['p2_y']:.4f},"
                            f"{m['distance']:.4f},{m['plane']}\n")
            self.statusBar().showMessage(f"CSVエクスポート完了: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"エクスポート失敗:\n{str(e)}")


# =============================================================================
# デモデータ
# =============================================================================
def generate_demo_data():
    print("デモデータ生成中...")
    nx = ny = nz = 80
    dx = dy = dz = 0.1
    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    positions = [
        (4.0, 4.0, 4.0), (4.0, 2.0, 2.0), (2.0, 4.0, 2.0), (2.0, 2.0, 4.0),
        (0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 4.0),
        (6.0, 4.0, 2.0), (6.0, 2.0, 4.0), (2.0, 6.0, 4.0), (4.0, 6.0, 2.0),
    ]
    sigma = 3.0
    for ax, ay, az in positions:
        cx, cy, cz = ax/dx, ay/dy, az/dz
        for zi in range(max(0,int(cz)-10), min(nz,int(cz)+11)):
            for yi in range(max(0,int(cy)-10), min(ny,int(cy)+11)):
                for xi in range(max(0,int(cx)-10), min(nx,int(cx)+11)):
                    r2 = (xi-cx)**2 + (yi-cy)**2 + (zi-cz)**2
                    volume[zi,yi,xi] += np.exp(-r2/(2*sigma**2))
    volume += np.random.normal(0, 0.01, volume.shape).astype(np.float32)
    volume = np.clip(volume, 0, None)
    atoms = [{'element':'Si','x':ax,'y':ay,'z':az} for ax,ay,az in positions]
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
        window.atoms = atoms
        window.lbl_xyz_info.setText(f"デモ: {len(atoms)}個のSi原子")
        window._update_atom_overlay()
        window.statusBar().showMessage("デモモードで起動")

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

 

