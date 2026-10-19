# AGENTS.md — MobileSAM 图像分割 GUI

## 项目概述

基于 **MobileSAM**（轻量级 Segment Anything）的交互式图像分割桌面应用，PySide6 单文件实现。

**核心流程**：打开 JPG 图片 → 用框/点提示选中目标 → 点击"开始分割" → 输出掩码、叠加图、裁剪图到 `output/` 目录。

**功能亮点**：
- 两种提示模式：**框提示**（默认，两点画矩形）与**点提示**（单点标记前景），`QRadioButton` 切换
- **延迟加载**：模型仅在首次点击"开始分割"时加载，启动零等待
- 权重自动下载（~40 MB，GitHub raw），CUDA 优先、CPU 回退
- 结果自动保存，无需手动导出

## Run / Build

```bash
uv run main.py          # 启动 GUI
uv add <package>        # 添加依赖
```

Python ≥3.14，包管理器为 `uv`。依赖声明在 `pyproject.toml`（`mobile-sam` 从 GitHub 源码安装）。

## 架构概览

- **`ClickableLabel(QLabel)`** — 交互核心。维护 `_points: list[QPoint]` 与 `_mode`（`"box"` | `"point"`）；点击后重绘并通过 `prompt_updated` 信号通知主窗口
- **`ImageViewer(QMainWindow)`** — 主窗口。负责打开图片、图像缩放、坐标变换、模型加载与分割执行
- **模型层** — `mobile_sam.sam_model_registry["vit_t"]` + `SamPredictor`，权重存于 `checkpoints/mobile_sam.pt`

### 提示交互规则
- **点提示**：单击记录 1 个前景点（label=1），再次单击替换
- **框提示**：两次单击定矩形对角；第三次单击清空重来
- 切换模式或打开新图会清空当前提示（`set_prompt_mode` / `clear_prompt`）
- "开始分割"按钮可用性由 `has_prompt()` 决定：点模式 ≥1 点、框模式 ≥2 点

### 坐标变换
`ClickableLabel` 在 **pixmap 显示坐标** 上操作（居中绘制，有偏移 `ox/oy`）。
`_to_original_coords()` / `_to_original_point()` 将显示坐标按缩放比映射到原始图像坐标。
**修改任何涉及坐标的逻辑时必须同步处理这两套坐标系。**

### 图像缩放
打开图片时自动缩放到屏幕的 50% (`SCALE_FACTOR = 0.5`)，保持宽高比。
`_original_pixmap` 存原始尺寸，`_display_pixmap` 存缩放后的显示尺寸。
窗口尺寸按显示图 + 控件占用空间（边距/按钮行/状态栏/窗口边框）自适应调整。

### 模型权重
- 本地路径: `checkpoints/mobile_sam.pt`（目录自动创建）
- 缺失时自动从 GitHub raw 下载（~40 MB）；失败则弹窗提示按 URL 手动下载
- 设备选择：`"cuda"` 优先，否则 `"cpu"`

### 分割流程 (`run_segmentation`)
1. 构建提示：点模式 → `point_coords` + `point_labels=[1]`；框模式 → `input_box`
2. 懒加载模型（若未加载，失败则弹 `QMessageBox.critical` 并中止）
3. `predictor.set_image(image_np, image_format="RGB")`（图像统一 `convert("RGB")`）
4. `predict(..., multimask_output=False)` 取单个最佳 mask（**bool 数组，H×W 无通道维**）
5. 保存三文件，状态栏显示 mask score

### 输出
分割结果输出到原图同目录的 `output/` 子目录（自动创建），包含三个文件：
- `{name}_mask.png` — 二值掩码（`mask * 255`）
- `{name}_overlay.png` — 半透明绿色叠加（原图与纯绿各 50% 混合）
- `{name}_crop.png` — 按 mask 包围盒裁剪的分割区域

## 常见陷阱

- **坐标偏移**: `ClickableLabel.paintEvent` 和 `mousePressEvent` 中的 `ox/oy` 计算必须一致，否则点击位置与绘制位置不匹配
- **警告抑制**: 文件顶部 `warnings.filterwarnings` 抑制了 `tiny_vit` 覆盖注册与 `timm` 导入弃用警告，不要移除
- **图像格式**: 仅支持 JPG/JPEG 打开；内部处理统一转为 RGB (`Image.open(...).convert("RGB")`)
- **mask 维度**: `masks[0]` 是 `(H, W)` 的 bool 数组，叠加时用布尔索引 `overlay[mask]`；`ys, xs = np.where(mask)` 用于包围盒裁剪
- **按钮状态**: 分割期间禁用"开始分割"防止重入；模型加载/分割异常需恢复按钮可用（`finally` 块）
- **下载失败处理**: `_load_model` 抛 `RuntimeError`（含手动下载指引），由 `run_segmentation` 捕获后弹窗，不要静默吞掉
