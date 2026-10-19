import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# 过滤 mobile_sam 与 timm 的兼容性警告（不影响功能）
warnings.filterwarnings("ignore", message=".*Overwriting tiny_vit.*")
warnings.filterwarnings("ignore", message=r".*Importing from timm\.models\..* is deprecated.*")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QRadioButton,
)
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtCore import Qt, Signal, QPoint


# ---------------------------------------------------------------------------
# 可点击的图像标签，支持点提示（单点）和框提示（矩形）
# ---------------------------------------------------------------------------
class ClickableLabel(QLabel):
    prompt_updated = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._points: list[QPoint] = []
        self._mode: str = "box"  # "box" | "point"

    # ---- 模式切换 ----
    def set_prompt_mode(self, mode: str) -> None:
        """切换提示模式：'point' 或 'box'"""
        self._mode = mode
        self._points.clear()
        self.update()
        self.prompt_updated.emit()

    # ---- 查询 ----
    def has_prompt(self) -> bool:
        if self._mode == "point":
            return len(self._points) >= 1
        return len(self._points) >= 2

    def clear_prompt(self) -> None:
        self._points.clear()
        self.update()
        self.prompt_updated.emit()

    def get_point_pixmap(self) -> tuple[int, int] | None:
        """点提示模式：返回 pixmap 坐标系下的 (x, y)"""
        if self._mode != "point" or not self._points:
            return None
        return (self._points[0].x(), self._points[0].y())

    def get_rect_pixmap(self) -> tuple[int, int, int, int] | None:
        """框提示模式：返回 pixmap 坐标系下的 (x1, y1, x2, y2)"""
        if self._mode != "box" or len(self._points) < 2:
            return None
        x1, y1 = self._points[0].x(), self._points[0].y()
        x2, y2 = self._points[1].x(), self._points[1].y()
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    # ---- 鼠标事件 ----
    def mousePressEvent(self, event) -> None:
        if self.pixmap() is None:
            return
        pm = self.pixmap()
        ox = (self.width() - pm.width()) // 2
        oy = (self.height() - pm.height()) // 2
        px = int(event.position().x() - ox)
        py = int(event.position().y() - oy)
        if px < 0 or py < 0 or px >= pm.width() or py >= pm.height():
            return

        if self._mode == "point":
            self._points = [QPoint(px, py)]
        else:
            if len(self._points) >= 2:
                self._points.clear()
            self._points.append(QPoint(px, py))

        self.update()
        self.prompt_updated.emit()

    # ---- 绘制 ----
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        pm = self.pixmap()
        if pm is None:
            return
        ox = (self.width() - pm.width()) // 2
        oy = (self.height() - pm.height()) // 2
        painter = QPainter(self)

        if self._mode == "point" and self._points:
            # 绘制十字准星
            x, y = self._points[0].x() + ox, self._points[0].y() + oy
            painter.setPen(QPen(QColor(0, 200, 0), 2))
            R = 8
            painter.drawLine(x - R, y, x + R, y)
            painter.drawLine(x, y - R, x, y + R)
            painter.drawEllipse(x - R, y - R, R * 2, R * 2)

        elif self._mode == "box" and len(self._points) >= 2:
            rect = self.get_rect_pixmap()
            if rect:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(QColor(255, 0, 0, 30))
                painter.drawRect(ox + rect[0], oy + rect[1],
                                 rect[2] - rect[0], rect[3] - rect[1])

        painter.end()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class ImageViewer(QMainWindow):
    CHECKPOINT_URL = (
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片查看器 - MobileSAM 分割")
        self.resize(400, 300)

        # —— 状态 ——
        self._original_pixmap: QPixmap | None = None   # 原始尺寸图像
        self._display_pixmap: QPixmap | None = None     # 缩放后显示用
        self._image_path: str | None = None
        self._predictor = None

        # —— UI ——
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 按钮行
        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("选择图片 (JPG)")
        self.btn_open.clicked.connect(self.open_image)
        btn_row.addWidget(self.btn_open)

        self.btn_segment = QPushButton("开始分割")
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self.run_segmentation)
        btn_row.addWidget(self.btn_segment)
        layout.addLayout(btn_row)

        # 提示模式单选框
        mode_row = QHBoxLayout()
        self.radio_box = QRadioButton("框提示")
        self.radio_box.setChecked(True)
        self.radio_point = QRadioButton("点提示")
        self.radio_box.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.radio_box)
        mode_row.addWidget(self.radio_point)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # 可点击图像标签
        self.image_label = ClickableLabel("请点击按钮选择一张 JPG 图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #f0f0f0; border: 1px dashed #aaa;"
        )
        self.image_label.setMinimumSize(200, 150)
        self.image_label.prompt_updated.connect(self._on_prompt_updated)
        layout.addWidget(self.image_label)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # 打开图片
    # ------------------------------------------------------------------
    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择一张 JPG 图片", "",
            "JPEG 图片 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_label.setText("无法加载该图片，请确认文件格式正确。")
            return

        self._image_path = file_path
        self._original_pixmap = pixmap
        self.image_label.clear_prompt()
        self.btn_segment.setEnabled(False)
        self.status_label.setText("")

        # 计算额外占用空间
        central_layout = self.centralWidget().layout()
        assert central_layout is not None
        btn_height = self.btn_open.sizeHint().height()
        margin = central_layout.contentsMargins()
        extra_w = margin.left() + margin.right()
        extra_h = (margin.top() + margin.bottom()
                   + central_layout.spacing() * 3 + btn_height
                   + self.status_label.sizeHint().height())
        frame_h = self.frameGeometry().height() - self.geometry().height()

        # 缩放因子
        SCALE_FACTOR = 0.5
        screen = self.screen().availableGeometry()
        MAX_W = int((screen.width() - extra_w) * SCALE_FACTOR)
        MAX_H = int((screen.height() - extra_h - frame_h) * SCALE_FACTOR)

        display = pixmap
        if pixmap.width() > MAX_W or pixmap.height() > MAX_H:
            s = min(MAX_W / pixmap.width(), MAX_H / pixmap.height())
            display = pixmap.scaled(
                int(pixmap.width() * s), int(pixmap.height() * s),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._display_pixmap = display

        self.image_label.setPixmap(display)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet("")

        win_w = display.width() + extra_w
        win_h = display.height() + extra_h + frame_h
        self.resize(win_w, win_h)

    # ------------------------------------------------------------------
    # 矩形框更新回调
    # ------------------------------------------------------------------
    def _on_prompt_updated(self) -> None:
        self.btn_segment.setEnabled(self.image_label.has_prompt())

    # ------------------------------------------------------------------
    # 提示模式切换
    # ------------------------------------------------------------------
    def _on_mode_changed(self) -> None:
        mode = "point" if self.radio_point.isChecked() else "box"
        self.image_label.set_prompt_mode(mode)

    # ------------------------------------------------------------------
    # 坐标还原：显示坐标 -> 原始图像坐标
    # ------------------------------------------------------------------
    def _to_original_coords(self, rect_pixmap: tuple[int, int, int, int]):
        """将 pixmap 显示坐标系的矩形转换为原始图像坐标系"""
        if self._original_pixmap is None or self._display_pixmap is None:
            return rect_pixmap
        sx = self._original_pixmap.width() / self._display_pixmap.width()
        sy = self._original_pixmap.height() / self._display_pixmap.height()
        return (
            int(rect_pixmap[0] * sx),
            int(rect_pixmap[1] * sy),
            int(rect_pixmap[2] * sx),
            int(rect_pixmap[3] * sy),
        )

    def _to_original_point(self, point_pm: tuple[int, int]) -> tuple[int, int]:
        """将 pixmap 显示坐标系的点转换为原始图像坐标系"""
        if self._original_pixmap is None or self._display_pixmap is None:
            return point_pm
        sx = self._original_pixmap.width() / self._display_pixmap.width()
        sy = self._original_pixmap.height() / self._display_pixmap.height()
        return (int(point_pm[0] * sx), int(point_pm[1] * sy))

    # ------------------------------------------------------------------
    # MobileSAM 分割
    # ------------------------------------------------------------------
    def _load_model(self):
        """延迟加载 MobileSAM 模型（仅第一次调用时）"""
        if self._predictor is not None:
            return

        self.status_label.setText("正在加载 MobileSAM 模型...")
        QApplication.processEvents()

        import torch
        from mobile_sam import sam_model_registry, SamPredictor

        # 检查 / 下载 checkpoint
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / "mobile_sam.pt"

        if not checkpoint_path.exists():
            self.status_label.setText("正在下载模型权重 (~40 MB)...")
            QApplication.processEvents()
            try:
                urllib.request.urlretrieve(
                    self.CHECKPOINT_URL, str(checkpoint_path)
                )
            except Exception as e:
                raise RuntimeError(
                    f"模型下载失败：{e}\n"
                    f"请手动从 {self.CHECKPOINT_URL} 下载 "
                    f"并放到 {checkpoint_dir}"
                )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self.status_label.setText(f"模型已加载（{device}）")

    def run_segmentation(self) -> None:
        is_point_mode = self.radio_point.isChecked()

        if is_point_mode:
            pt_pm = self.image_label.get_point_pixmap()
            if pt_pm is None or self._image_path is None:
                return
        else:
            rect_pm = self.image_label.get_rect_pixmap()
            if rect_pm is None or self._image_path is None:
                return

        try:
            self._load_model()
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))
            self.status_label.setText("模型加载失败")
            return
        assert self._predictor is not None

        # 构建提示参数
        if is_point_mode:
            pt_pm = self.image_label.get_point_pixmap()
            assert pt_pm is not None
            ox, oy = self._to_original_point(pt_pm)
            point_coords = np.array([[ox, oy]])
            point_labels = np.array([1])  # 1=前景
            input_box = None
        else:
            rect_pm = self.image_label.get_rect_pixmap()
            assert rect_pm is not None
            x1, y1, x2, y2 = self._to_original_coords(rect_pm)
            input_box = np.array([x1, y1, x2, y2])
            point_coords = None
            point_labels = None

        self.status_label.setText("正在分割...")
        self.btn_segment.setEnabled(False)
        QApplication.processEvents()

        try:
            image_np = np.array(Image.open(self._image_path).convert("RGB"))
            self._predictor.set_image(image_np, image_format="RGB")

            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box,
                multimask_output=False,
            )
            mask = masks[0]  # bool, (H, W)

            # 保存结果到原图同目录
            base = Path(self._image_path).stem
            out_dir = Path(self._image_path).parent / "output"
            out_dir.mkdir(exist_ok=True)

            # 1) 二值 mask
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(str(out_dir / f"{base}_mask.png"))

            # 2) 叠加可视化
            overlay = image_np.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0],
                                                             dtype=np.uint8) * 0.5
            Image.fromarray(overlay).save(str(out_dir / f"{base}_overlay.png"))

            # 3) 裁剪分割区域
            ys, xs = np.where(mask)
            if len(ys) > 0:
                crop = image_np[ys.min():ys.max()+1, xs.min():xs.max()+1]
                Image.fromarray(crop).save(str(out_dir / f"{base}_crop.png"))

            self.status_label.setText(
                f"分割完成！结果保存在 {out_dir} (score={scores[0]:.3f})"
            )
        except Exception as e:
            QMessageBox.critical(self, "分割失败", str(e))
            self.status_label.setText("分割失败")
        finally:
            self.btn_segment.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())

