# Atom Image Viewer — CLAUDE.md

## プロジェクト概要
`atomic_image_reconstruction_viewer_v3.py` — PyQt5 + Matplotlib による 3D 原子像再構成ビューア。
3D-AIR-IMAGE API または .npy / .npz / .xml_air ファイルからボリュームデータを読み込み、
XY / XZ / YZ スライス表示・2点距離/3点角度測定・自動ピーク検出を行う。

## 主要ファイル
| ファイル | 用途 |
|---|---|
| `atomic_image_reconstruction_viewer_v3.py` | メインアプリケーション (単一ファイル) |
| `api_3d_air_image_qt6.py` | 3D-AIR-IMAGE API クライアント (オプション) |

## アーキテクチャ

### モジュールレベル (クラス外)
- `PEAK_REFINE_*` 定数 (0=Gaussian, 1=Centroid, 2=Parabola, 3=OFF)
- `_peak_find_coarse(data_2d, px, py, sr)` → `(ix, iy)` 粗最大値
- `peak_refine_parabola_3d(vol, iz, iy, ix)` → `(x,y,z,log)`
- `peak_refine_centroid_3d(vol, iz, iy, ix, fit_size)` → `(x,y,z,log)`
- `peak_refine_gaussian_3d(vol, iz, iy, ix, fit_size)` → `(x,y,z,sx,sy,sz,log) | None`
- `_apply_peak_refinement_on_region(vol, iz, iy, ix, mode, fit_size)` → `(x,y,z)`

### クラス構造
```
AirWindowSelectDialog   3D-AIR-IMAGE ウィンドウ選択ダイアログ
SliceCanvas             matplotlib FigureCanvas サブクラス
PeakDetectionWorker     QThread — 非同期ピーク検出
AtomViewerWindow        QMainWindow — メインウィンドウ
```

### AtomViewerWindow 主要状態
| 属性 | 型 | 説明 |
|---|---|---|
| `self.volume` | ndarray (Nz,Ny,Nx) float64 | ボリュームデータ |
| `self.metadata` | dict | voxel_size, x/y/z_range |
| `self.atom_sets` | list[dict] | 読み込んだXYZセット一覧 |
| `self.active_atom_set_index` | int | 現在編集中のセット (-1=なし) |
| `self.click_points` | list[dict] | 進行中の選択点 (距離・角度共用) |
| `self.measurements` | list[dict] | 確定済み距離履歴 |
| `self.angle_measurements` | list[dict] | 確定済み角度履歴 |
| `self.detected_peaks` | list[dict] | 自動検出ピーク |
| `self.measure_type` | str | 'distance' or 'angle' |
| `self.spin_cluster_pos` | dict[str, QDoubleSpinBox] | CLUSTER SLICE 独立スピン (keys: 'x','y','z') |
| `self.slider_cluster` | dict[str, QSlider] | CLUSTER SLICE 独立スライダー (keys: 'x','y','z') |

### atom_set 辞書構造
```python
{
    'filepath': Path | None,    # ファイルパス (デモは None)
    'name': str,                # 表示名 (ファイルステム)
    'atoms': list[dict],        # load_xyz の戻り値
    'visible': bool,            # 表示ON/OFF
    'offset': [ox, oy, oz],     # 原子全体のシフト量 (Å, 表示のみ)
    'element_colors': dict,     # 元素→色文字列 '#rrggbb'
    'element_radii': dict,      # 元素→半径 float (Å)
}
```

## 座標系の一致 (重要原則)
> XYZファイルの原子座標とホログラムの Å 座標は **同じ原点・スケール・向き** である。
> `offset = [0,0,0]` の状態で原子位置がホログラムのピーク位置と一致するのが正しい状態。
> ATOM OFFSET はあくまで微調整用であり、座標系の不一致を埋めるために使うものではない。

## 座標系
- ボリューム配列: `volume[z, y, x]`
- Å 座標: **X/Y/Z すべて中心原点** (ボリューム中心 = 0 Å)
- `metadata['x_range'] = (-nx*dx/2, nx*dx/2)` など (Z も同様)
- スライス番号→Å変換: `ang = (slider_value - N//2) * d`
- Å→スライス番号: `_angstrom_to_slice_idx(axis, ang)` を使用
- `spin_z/y/x_origin` は廃止 (常に中心固定)。スライス位置は `spin_z/y/x` が Å 値を直接保持

## XYZファイル管理
- `QFileDialog.getOpenFileNames` で複数ファイルを一度に読み込み
- `list_xyz_files` (QListWidget+チェックボックス) でファイルごとに表示/非表示切替
- 行選択で `active_atom_set_index` が変わり、ELEMENTS UI と ATOM OFFSET UI が切り替わる
- 削除ボタン・全表示/全非表示ボタン付き
- 廃止: `btn_xyz_prev`, `btn_xyz_next`, `xyz_file_list`, `xyz_file_index`, `_load_xyz_by_index`

## ATOM OFFSET 機能
- 各XYZファイルが独立に `offset = [ox, oy, oz]` を持つ
- 「▸ ATOM OFFSET (アクティブファイル)」グループで編集
- `spin_offset_x/y/z`: -100〜+100 Å, step=0.05
- ファイルリスト切り替え時に各スピンボックスを対応セットの offset に同期 (blockSignals)
- 元データ (`atom_set['atoms']`) は書き換えない。描画時にオフセットを加算するのみ
- **平面タブに応じた軸制約**: 現在の平面で意味のある法線方向の軸のみ有効化
  - XY平面タブ → Z オフセットのみ有効 (X/Y は無効化)
  - XZ平面タブ → Y オフセットのみ有効 (X/Z は無効化)
  - YZ平面タブ → X オフセットのみ有効 (Y/Z は無効化)
  - `_update_offset_axis_enabled()` が `slice_tab.currentChanged` + `_update_active_ui` から呼ばれる

## ピーク探索フロー (クリック時)
1. `combo_peak_refine` が OFF → 旧パス: `_snap_or_find_peak` → `_find_peak` (3×3重心)
2. OFF 以外 → `_find_peak_3d_refined`: スナップ → `_peak_find_coarse` → 選択アルゴリズム
3. ガウシアン失敗時はパラボラにフォールバック
4. `chk_fit_log` ON でステータスバーにデバッグ情報を表示

## 実装済み機能
- ① サブボクセル精度 (Gaussian/Centroid/Parabola/OFF)
  - 自動ピーク検出にも適用 (100件以上は進捗表示)
  - PeakDetectionWorker (スレッド) にも refine_mode 渡し済み
- ② 選択リセット (各タブの赤ボタン `btn_reset_dist` / `btn_reset_angle`)
- ③ ボリューム再描画 `btn_redraw_volume` (データ・履歴維持)
- ④ 3点角度測定のZスライス追従
  - `_redraw_measurements` で確定済み角度測定を **保存済みÅ座標** から直接再描画
  - 距離測定は引き続き `_find_peak` でスライス再探索; 角度は座標固定 (3D測定のため)
- ⑤ Z/Y/X 中心原点座標系 (XZ/YZ 縦軸も -N〜+N Å)
- ⑥ 複数XYZファイル保持・個別表示切替
- ⑦ ファイルごとの原子位置オフセット (X/Y/Z 独立、平面タブで軸制約)
- ⑧ 原子オーバーレイ表示判定: 法線方向座標が `|Δ| ≤ voxel_size/2` の原子のみ表示
  - XY平面: `|atom.z + oz - z_slice| ≤ dz/2`、描画座標は `(atom.x+ox, atom.y+oy)`
  - XZ平面: `|atom.y + oy - y_slice| ≤ dy/2`、描画座標は `(atom.x+ox, atom.z+oz)`
  - YZ平面: `|atom.x + ox - x_slice| ≤ dx/2`、描画座標は `(atom.y+oy, atom.z+oz)`
- ⑨ クラスタースライス位置の独立制御 (CLUSTER SLICE グループ)
  - `chk_cluster_link_hologram` で連動/独立切替 (デフォルト連動)
  - `_current_cluster_pos_angstrom(axis)` が判定基準位置を返す
  - 独立モード時は `spin_cluster_pos[axis]` / `slider_cluster[axis]` で基準位置を指定
  - `btn_cluster_sync`: 現在のホログラム位置をクラスタースピンにコピー
- ⑩ スライス位置UIをÅ単位に統一
  - `spin_z/y/x` を `QDoubleSpinBox` (Å値) に変更; スライス番号・原点スピン廃止
  - 配列インデックスは `_angstrom_to_slice_idx(axis, ang)` で都度変換
  - タブタイトル簡素化: `"XY  ·  Z = {z_pos:+.3f} Å"` 形式 (slice番号表記なし)

## 開発ノート
- `_find_peak(self, data_2d, px, py, search_r)` は距離測定の `_redraw_measurements` 用に保持
  (マーカー再描画専用、精度は問わない)
- 角度測定の `_redraw_measurements` は `_find_peak` を使わず `ameas['p1_x']` 等を直接参照
- `spin_fit_size` は奇数強制 (`_on_fit_size_changed` で偶数→偶数+1)
- ボリューム描画はキャッシュなし: `_update_slice()` が毎回 numpy 配列から直接 `imshow`
- `self.atoms` は `@property` (後方互換): active set の atoms を返す
- デモモードの原子座標は中心原点 (絶対位置 - 4.0 Å)
- **描画座標と判定座標の分離原則**:
  - XY平面: 判定=`atom.z + oz`、描画=`(atom.x + ox, atom.y + oy)`
  - XZ平面: 判定=`atom.y + oy`、描画=`(atom.x + ox, atom.z + oz)`
  - YZ平面: 判定=`atom.x + ox`、描画=`(atom.y + oy, atom.z + oz)`
  - 法線方向のオフセットは判定にのみ、面内オフセットは描画にのみ使う。混ぜない。
- `_current_slice_pos_angstrom(axis)`: ホログラム現在位置 (Å) を返す → `spin_z/y/x.value()` の薄いラッパ
- `_current_cluster_pos_angstrom(axis)`: 原子オーバーレイ判定基準 (Å)。連動時=ホログラム位置、独立時=クラスタースピン値
- `_angstrom_to_slice_idx(axis, ang)`: Å値→最寄りスライスインデックス変換
- SLICE スピンは `QDoubleSpinBox` (Å単位)。スライダーは int (スライスインデックス) のまま双方向同期
- スライダー↔スピン同期: `_on_slider_z/y/x_changed` / `_on_spin_z/y/x_changed` がblockSignalsで循環防止

## 実行方法
```bash
python atomic_image_reconstruction_viewer_v3.py          # 通常起動
python atomic_image_reconstruction_viewer_v3.py --demo   # デモデータで起動
```
