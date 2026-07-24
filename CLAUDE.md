# Atom Image Viewer — CLAUDE.md

## プロジェクト概要
`atomic_image_reconstruction_viewer.py` — PyQt5 + Matplotlib による 3D 原子像再構成ビューア。
3D-AIR-IMAGE API または .npy / .npz / .xml_air ファイルからボリュームデータを読み込み、
XY スライス表示・2点距離/3点角度測定・自動ピーク検出を行う。

## 主要ファイル
| ファイル | 用途 |
|---|---|
| `atomic_image_reconstruction_viewer.py` | メインアプリケーション (単一ファイル) |
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
| `self.click_points` | list[dict] | 進行中の選択点 (距離・角度共用) |
| `self.measurements` | list[dict] | 確定済み距離履歴 |
| `self.angle_measurements` | list[dict] | 確定済み角度履歴 |
| `self.detected_peaks` | list[dict] | 自動検出ピーク |
| `self.measure_type` | str | 'distance' or 'angle' |

## 座標系
- ボリューム配列: `volume[z, y, x]`
- Å 座標: X/Y は中心原点、Z は 0 始まり
- `metadata['x_range'] = (-nx*dx/2, nx*dx/2)` など

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
  - 進行中の角度クリック点 (`click_points`) も保存済みÅ座標で追従
  - 距離測定は引き続き `_find_peak` でスライス再探索; 角度は座標固定 (3D測定のため)

## 開発ノート
- `_find_peak(self, data_2d, px, py, search_r)` は距離測定の `_redraw_measurements` 用に保持
  (マーカー再描画専用、精度は問わない)
- 角度測定の `_redraw_measurements` は `_find_peak` を使わず `ameas['p1_x']` 等を直接参照
  (Z変更で別ピークを拾う問題を回避; 角度値は確定済みなので再計算不要)
- `spin_fit_size` は奇数強制 (`_on_fit_size_changed` で偶数→偶数+1)
- ボリューム描画はキャッシュなし: `_update_slice()` が毎回 numpy 配列から直接 `imshow`

## 実行方法
```bash
python atomic_image_reconstruction_viewer.py          # 通常起動
python atomic_image_reconstruction_viewer.py --demo   # デモデータで起動
```
