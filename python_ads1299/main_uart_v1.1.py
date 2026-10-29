import csv
import datetime
import sys
from numbers import Real

import numpy as np
import serial
from serial.tools import list_ports
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ads1299 import ADS1298Data, ADS1298cmd


class MainWindow(QMainWindow):
    """ADS1299八通道实时波形显示主窗口。"""

    def __init__(self):
        super().__init__()

        # 先初始化运行状态，再创建界面和定时器。
        self.AdsData = ADS1298Data()
        self.serial_port = None
        self.is_collecting = False
        self.channel_states = [True] * 8

        # 波形显示参数。
        self.display_seconds = 10.0
        self.max_display_points = 3000

        # 每个通道独立自动纵轴量程。
        # 初始显示为 +/-100 uV；收到数据后，根据该通道最近时间窗内
        # 的实际幅值自动调整。放大立即执行，缩小延迟执行，避免画面抖动。
        self.initial_y_half_range = 100.0
        self.min_y_half_range = 10.0
        self.autoscale_padding = 1.15
        self.autoscale_shrink_ratio = 0.55
        self.autoscale_shrink_delay = 12
        self.channel_half_ranges = [self.initial_y_half_range] * 8
        self.channel_shrink_counts = [0] * 8

        # 通道名称及固定颜色。
        self.channel_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
        self.channel_colors = [
            "#2F80FF",  # Fp1 蓝色
            "#FF7A18",  # Fp2 橙色
            "#F04A9B",  # C3 粉色
            "#20C878",  # C4 绿色
            "#8D6BEB",  # P3 紫色
            "#32B7E9",  # P4 青蓝色
            "#E3A600",  # O1 黄色
            "#12B8B5",  # O2 青色
        ]

        self.lines = []
        self.channel_buttons = []
        self.channel_color_blocks = []
        self.y_top_labels = []
        self.y_bottom_labels = []
        self.channel_name_labels = []

        # 响应式字体和控件尺寸。窗口缩放或切换到不同分辨率屏幕时自动更新。
        self.ui_scale = 1.0
        self.ui_font_px = 15
        self.small_font_px = 13
        self.plot_scale_font_px = 11
        self.plot_channel_font_px = 13
        self.control_height = 38
        self.checkbox_height = 34
        self.checkbox_indicator_size = 17
        self.color_block_size = 20

        self.initUI()

    def initUI(self):
        """创建窗口、控制区和八通道波形区。"""
        self.setWindowTitle("ADS1299 八通道脑电实时显示")
        self.setGeometry(80, 80, 1600, 900)
        self.setMinimumSize(1200, 720)

        # 根据当前屏幕分辨率、DPI和窗口大小计算字体缩放比例。
        self._update_scaled_metrics()
        self._apply_responsive_style()

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # ---------------- 左侧控制区 ----------------
        left_vbox = QVBoxLayout()
        left_vbox.setAlignment(Qt.AlignTop)
        left_vbox.setSpacing(12)
        left_vbox.addSpacerItem(QSpacerItem(220, 10, QSizePolicy.Fixed, QSizePolicy.Fixed))

        port_row = QHBoxLayout()
        port_label = QLabel("选择串口")
        self.port_combo = QComboBox(self)
        port_row.addWidget(port_label)
        port_row.addWidget(self.port_combo, 1)

        self.btn_refresh = QPushButton("刷新串口", self)
        self.btn_refresh.clicked.connect(self.refresh_ports)

        self.btn_open = QPushButton("打开串口", self)
        self.btn_open.clicked.connect(self.open_serial)

        self.btn_start = QPushButton("开始采集", self)
        self.btn_start.clicked.connect(self.start_collect)
        self.btn_start.setEnabled(False)

        self.btn_export_csv = QPushButton("导出数据", self)
        self.btn_export_csv.clicked.connect(self.export_csv)

        channel_group = QGroupBox("通道选择")
        channel_layout = QVBoxLayout()
        channel_layout.setSpacing(8)

        for i in range(8):
            row = QHBoxLayout()
            row.setSpacing(8)

            color_block = QLabel(self)
            color_block.setFixedSize(self.color_block_size, self.color_block_size)
            self._set_color_block_style(i, enabled=True, widget=color_block)

            checkbox = QCheckBox(self.channel_names[i], self)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(self._channel_checkbox_style(i))
            checkbox.toggled.connect(
                lambda state, idx=i: self.toggle_channel(state, idx)
            )

            row.addWidget(color_block)
            row.addWidget(checkbox, 1)
            channel_layout.addLayout(row)

            self.channel_color_blocks.append(color_block)
            self.channel_buttons.append(checkbox)

        channel_group.setLayout(channel_layout)

        left_vbox.addLayout(port_row)
        left_vbox.addWidget(self.btn_refresh)
        left_vbox.addWidget(self.btn_open)
        left_vbox.addWidget(self.btn_start)
        left_vbox.addWidget(self.btn_export_csv)
        left_vbox.addWidget(channel_group)
        left_vbox.addStretch(1)

        # ---------------- 右侧波形区 ----------------
        right_vbox = QVBoxLayout()
        right_vbox.setContentsMargins(0, 0, 0, 0)

        self.fig, self.axes = plt.subplots(
            8,
            1,
            sharex=True,
            figsize=(16, 9),
            gridspec_kw={"hspace": 0.0},
        )
        self.fig.patch.set_facecolor("#071426")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #071426; border: none;")

        self._init_waveform_axes()
        right_vbox.addWidget(self.canvas, 1)

        root_layout.addLayout(left_vbox, 0)
        root_layout.addLayout(right_vbox, 1)

        central_widget = QWidget(self)
        central_widget.setLayout(root_layout)
        self.setCentralWidget(central_widget)

        # 每100 ms读取串口并刷新波形。
        self.update_show_timer = QTimer(self)
        self.update_show_timer.timeout.connect(self.update_show)
        self.update_show_timer.start(100)

        self.refresh_ports()

        # 窗口尺寸变化时延迟刷新字体，避免拖动窗口过程中频繁重绘。
        self.font_resize_timer = QTimer(self)
        self.font_resize_timer.setSingleShot(True)
        self.font_resize_timer.timeout.connect(self._refresh_responsive_fonts)

    def _calculate_ui_scale(self):
        """根据屏幕分辨率、逻辑DPI和当前窗口大小计算缩放比例。"""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return 1.0

        geometry = screen.availableGeometry()
        screen_width = max(1, geometry.width())
        screen_height = max(1, geometry.height())
        logical_dpi = max(72.0, float(screen.logicalDotsPerInch()))

        # 以1600x900、96 DPI为设计基准。DPI使用平方根补偿，
        # 可避免系统本身已开启缩放时字体被重复放大。
        screen_ratio = min(screen_width / 1600.0, screen_height / 900.0)
        dpi_ratio = (logical_dpi / 96.0) ** 0.5

        # 窗口缩小时字体适度缩小，放大窗口时字体同步增大。
        if self.width() > 200 and self.height() > 200:
            window_ratio = min(self.width() / 1600.0, self.height() / 900.0)
        else:
            window_ratio = screen_ratio

        combined = min(screen_ratio, max(0.85, window_ratio)) * dpi_ratio
        return float(np.clip(combined, 0.95, 1.85))

    def _update_scaled_metrics(self):
        """更新字体、按钮高度、色块尺寸等响应式参数。"""
        self.ui_scale = self._calculate_ui_scale()
        self.ui_font_px = max(15, round(16 * self.ui_scale))
        self.small_font_px = max(13, round(14 * self.ui_scale))
        self.plot_scale_font_px = max(11, round(12 * self.ui_scale))
        self.plot_channel_font_px = max(13, round(14 * self.ui_scale))
        self.control_height = max(38, round(40 * self.ui_scale))
        self.checkbox_height = max(34, round(36 * self.ui_scale))
        self.checkbox_indicator_size = max(17, round(18 * self.ui_scale))
        self.color_block_size = max(20, round(21 * self.ui_scale))

    def _apply_responsive_style(self):
        """把响应式字体应用到全部Qt控件和Matplotlib标签。"""
        font = QFont("Microsoft YaHei")
        font.setPixelSize(self.ui_font_px)
        QApplication.instance().setFont(font)
        self.setFont(font)

        radius = max(5, round(6 * self.ui_scale))
        horizontal_padding = max(8, round(10 * self.ui_scale))
        group_margin = max(12, round(14 * self.ui_scale))

        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: #071426;
                color: #DDE7F2;
                font-family: 'Microsoft YaHei';
                font-size: {self.ui_font_px}px;
            }}
            QLabel {{
                font-size: {self.ui_font_px}px;
            }}
            QGroupBox {{
                border: 1px solid #27405D;
                border-radius: {radius}px;
                margin-top: {group_margin}px;
                padding-top: {max(10, round(12 * self.ui_scale))}px;
                font-size: {self.ui_font_px}px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {horizontal_padding}px;
                padding: 0 {max(4, round(5 * self.ui_scale))}px;
            }}
            QComboBox, QPushButton {{
                min-height: {self.control_height}px;
                border: 1px solid #355574;
                border-radius: {radius}px;
                padding: 2px {horizontal_padding}px;
                background-color: #10243B;
                color: #EAF2FA;
                font-size: {self.ui_font_px}px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #10243B;
                color: #EAF2FA;
                selection-background-color: #24517A;
                font-size: {self.ui_font_px}px;
                min-height: {self.control_height}px;
            }}
            QPushButton:hover {{
                background-color: #173451;
                border-color: #4E7AA3;
            }}
            QPushButton:pressed {{
                background-color: #0B1C30;
            }}
            QPushButton:disabled, QComboBox:disabled {{
                color: #687A8C;
                background-color: #0B1828;
                border-color: #23364A;
            }}
            QMessageBox QLabel {{
                min-width: {max(320, round(360 * self.ui_scale))}px;
                font-size: {self.ui_font_px}px;
            }}
            QMessageBox QPushButton {{
                min-width: {max(90, round(100 * self.ui_scale))}px;
            }}
            """
        )

        # 复选框样式包含通道颜色，需要逐个重新应用。
        for i, checkbox in enumerate(getattr(self, "channel_buttons", [])):
            checkbox.setStyleSheet(self._channel_checkbox_style(i))

        for i, color_block in enumerate(getattr(self, "channel_color_blocks", [])):
            color_block.setFixedSize(self.color_block_size, self.color_block_size)
            self._set_color_block_style(i, self.channel_states[i], color_block)

        # 更新Matplotlib中的通道名、量程和时间轴字体。
        for label in getattr(self, "y_top_labels", []):
            label.set_fontsize(self.plot_scale_font_px)
        for label in getattr(self, "y_bottom_labels", []):
            label.set_fontsize(self.plot_scale_font_px)
        for label in getattr(self, "channel_name_labels", []):
            label.set_fontsize(self.plot_channel_font_px)

        if hasattr(self, "axes") and len(self.axes) > 0:
            self.axes[-1].tick_params(
                axis="x",
                which="major",
                labelsize=self.plot_scale_font_px,
                pad=max(6, round(7 * self.ui_scale)),
            )
            left_margin = float(np.clip(0.085 + 0.025 * (self.ui_scale - 1.0), 0.085, 0.135))
            bottom_margin = float(np.clip(0.055 + 0.015 * (self.ui_scale - 1.0), 0.055, 0.085))
            self.fig.subplots_adjust(
                left=left_margin,
                right=0.995,
                top=0.995,
                bottom=bottom_margin,
                hspace=0.0,
            )

        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _refresh_responsive_fonts(self):
        """窗口大小稳定后重新计算并应用字体。"""
        previous_scale = self.ui_scale
        self._update_scaled_metrics()
        if abs(self.ui_scale - previous_scale) < 0.03:
            return
        self._apply_responsive_style()

    def resizeEvent(self, event):
        """窗口缩放时触发响应式字体更新。"""
        super().resizeEvent(event)
        if hasattr(self, "font_resize_timer"):
            self.font_resize_timer.start(160)

    def _channel_checkbox_style(self, channel_idx):
        """生成带通道固定色的复选框样式。"""
        color = self.channel_colors[channel_idx]
        return f"""
            QCheckBox {{
                min-height: {self.checkbox_height}px;
                padding: 2px {max(8, round(9 * self.ui_scale))}px;
                border-radius: 5px;
                color: #8393A5;
                background-color: #0C1B2D;
                border: 1px solid #263D55;
                font-size: {self.small_font_px}px;
            }}
            QCheckBox:checked {{
                color: #FFFFFF;
                background-color: {color};
                border: 1px solid #FFFFFF;
                font-weight: bold;
            }}
            QCheckBox::indicator {{
                width: {self.checkbox_indicator_size}px;
                height: {self.checkbox_indicator_size}px;
            }}
        """

    def _set_color_block_style(self, channel_idx, enabled, widget=None):
        """设置通道固定色块的高亮或弱化状态。"""
        target = widget if widget is not None else self.channel_color_blocks[channel_idx]
        color = self.channel_colors[channel_idx]
        if enabled:
            target.setStyleSheet(
                f"background-color: {color}; border: 2px solid #FFFFFF; border-radius: 4px;"
            )
        else:
            target.setStyleSheet(
                f"background-color: {color}; border: 1px solid #3F5266; border-radius: 4px;"
            )

    def _init_waveform_axes(self):
        """初始化8个独立但共享时间轴的脑电波形子图。"""
        for i, ax in enumerate(self.axes):
            ax.set_facecolor("#0A1729")
            half_range = self.channel_half_ranges[i]
            ax.set_ylim(-half_range, half_range)
            ax.set_yticks([])
            ax.set_axisbelow(True)

            # 1秒主网格、0.2秒次网格。
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax.grid(
                True,
                which="major",
                axis="x",
                color="#183452",
                linestyle="-",
                linewidth=0.55,
                alpha=0.80,
            )
            ax.grid(
                True,
                which="minor",
                axis="x",
                color="#102943",
                linestyle="-",
                linewidth=0.35,
                alpha=0.55,
            )

            # 每行零电位参考线。
            ax.axhline(0.0, color="#112A45", linewidth=0.50, alpha=0.70)

            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#50647A")
            ax.spines["bottom"].set_linewidth(0.65)

            line, = ax.plot(
                [],
                [],
                color=self.channel_colors[i],
                linewidth=0.75,
                alpha=1.0,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            self.lines.append(line)

            # 左侧幅值参考和通道名称。纵轴数值会随该通道自动量程更新。
            top_label = ax.text(
                -0.028,
                0.80,
                f"+{self._format_scale_value(half_range)}",
                transform=ax.transAxes,
                color="#9EACBA",
                fontsize=self.plot_scale_font_px,
                ha="right",
                va="center",
                clip_on=False,
            )
            bottom_label = ax.text(
                -0.028,
                0.20,
                f"-{self._format_scale_value(half_range)}",
                transform=ax.transAxes,
                color="#758596",
                fontsize=self.plot_scale_font_px,
                ha="right",
                va="center",
                clip_on=False,
            )
            self.y_top_labels.append(top_label)
            self.y_bottom_labels.append(bottom_label)

            channel_label = ax.text(
                -0.006,
                0.50,
                self.channel_names[i],
                transform=ax.transAxes,
                color="#E6EDF5",
                fontsize=self.plot_channel_font_px,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )
            self.channel_name_labels.append(channel_label)

            if i < 7:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False,
                )
            else:
                ax.tick_params(
                    axis="x",
                    which="major",
                    colors="#C4CFDB",
                    labelsize=self.plot_scale_font_px,
                    length=0,
                    pad=6,
                )
                ax.tick_params(axis="x", which="minor", length=0)

        self.axes[-1].xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:.1f}s")
        )
        self.axes[-1].set_xlim(0.0, self.display_seconds)

        self.fig.subplots_adjust(
            left=0.085,
            right=0.995,
            top=0.995,
            bottom=0.055,
            hspace=0.0,
        )

    @staticmethod
    def _format_scale_value(value):
        """把纵轴半量程格式化为紧凑的显示文本。"""
        value = float(value)
        if value >= 100.0:
            return f"{value:.0f}"
        if value >= 10.0:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _nice_axis_limit(self, value):
        """将目标量程向上取整为适合显示的1/1.25/1.5/2/2.5等刻度。"""
        value = max(float(value), self.min_y_half_range)
        if not np.isfinite(value):
            return self.initial_y_half_range

        exponent = np.floor(np.log10(value))
        scale = 10.0 ** exponent
        fraction = value / scale
        nice_steps = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
        nice_fraction = next((step for step in nice_steps if fraction <= step), 10.0)
        return max(self.min_y_half_range, nice_fraction * scale)

    def _update_channel_y_scale(self, channel_idx, y_data):
        """根据当前通道显示窗口内的信号幅值，独立更新该通道纵轴。"""
        finite_data = np.asarray(y_data, dtype=np.float64)
        finite_data = finite_data[np.isfinite(finite_data)]
        if finite_data.size == 0:
            return

        # 高通后的脑电通常以0为中心，因此采用对称纵轴。
        # 99.5百分位降低孤立异常点的影响，最大值项又保证真实尖峰不会被截断。
        absolute_data = np.abs(finite_data)
        robust_peak = float(np.percentile(absolute_data, 99.5))
        absolute_peak = float(np.max(absolute_data))
        target_peak = max(
            robust_peak * self.autoscale_padding,
            absolute_peak * 1.05,
            self.min_y_half_range,
        )
        target_range = self._nice_axis_limit(target_peak)
        current_range = self.channel_half_ranges[channel_idx]

        # 幅值接近上边界时立即扩大量程，避免波形顶到边界。
        if target_range > current_range * 0.92:
            new_range = target_range
            self.channel_shrink_counts[channel_idx] = 0
        # 幅值长期显著减小时才缩小，避免纵轴频繁跳动。
        elif target_range < current_range * self.autoscale_shrink_ratio:
            self.channel_shrink_counts[channel_idx] += 1
            if self.channel_shrink_counts[channel_idx] >= self.autoscale_shrink_delay:
                new_range = target_range
                self.channel_shrink_counts[channel_idx] = 0
            else:
                return
        else:
            self.channel_shrink_counts[channel_idx] = 0
            return

        if np.isclose(new_range, current_range):
            return

        self.channel_half_ranges[channel_idx] = new_range
        self.axes[channel_idx].set_ylim(-new_range, new_range)
        value_text = self._format_scale_value(new_range)
        self.y_top_labels[channel_idx].set_text(f"+{value_text}")
        self.y_bottom_labels[channel_idx].set_text(f"-{value_text}")

    def refresh_ports(self):
        """扫描电脑上的可用串口并更新下拉框。"""
        current_selection = self.port_combo.currentText()
        self.port_combo.clear()

        ports = list_ports.comports()
        if not ports:
            self.port_combo.addItem("未检测到串口")
            self.port_combo.setEnabled(False)
            self.btn_open.setEnabled(False)
            return

        for port in sorted(ports):
            self.port_combo.addItem(port.device, port.description)

        self.port_combo.setEnabled(True)
        self.btn_open.setEnabled(True)

        available_names = [
            self.port_combo.itemText(i) for i in range(self.port_combo.count())
        ]
        if current_selection in available_names:
            self.port_combo.setCurrentText(current_selection)
        else:
            self.port_combo.setCurrentIndex(0)

    def open_serial(self):
        """打开或关闭当前选择的串口。"""
        if self.serial_port is None or not self.serial_port.is_open:
            port_name = self.port_combo.currentText()
            try:
                self.serial_port = serial.Serial(
                    port=port_name,
                    baudrate=1_000_000,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.5,
                )
            except serial.SerialException as exc:
                self.serial_port = None
                QMessageBox.critical(self, "串口错误", f"打开串口失败：\n{exc}")
                return

            self.btn_open.setText("关闭串口")
            self.btn_start.setEnabled(True)
            self.port_combo.setEnabled(False)
            self.btn_refresh.setEnabled(False)
            print(f"打开串口成功：{self.serial_port.name}")
            return

        self.stop_collection()
        try:
            self.serial_port.close()
        except serial.SerialException as exc:
            QMessageBox.warning(self, "串口提示", f"关闭串口时发生错误：\n{exc}")
        finally:
            self.serial_port = None
            self.btn_open.setText("打开串口")
            self.btn_start.setText("开始采集")
            self.btn_start.setEnabled(False)
            self.port_combo.setEnabled(True)
            self.btn_refresh.setEnabled(True)

    def start_collect(self):
        """切换开始或停止采集状态。"""
        if self.serial_port is None or not self.serial_port.is_open:
            QMessageBox.warning(self, "提示", "请先打开串口。")
            return

        if not self.is_collecting:
            self.AdsData.clear()
            self._clear_waveform_lines()
            try:
                self.serial_port.write(bytes(ADS1298cmd.start_collect_cmd()))
            except serial.SerialException as exc:
                QMessageBox.critical(self, "串口错误", f"开始采集命令发送失败：\n{exc}")
                return

            self.is_collecting = True
            self.btn_start.setText("停止采集")
        else:
            self.stop_collection()

    def stop_collection(self):
        """停止采集，但保留当前缓存数据用于显示和导出。"""
        if self.serial_port is not None and self.serial_port.is_open:
            try:
                self.serial_port.write(bytes(ADS1298cmd.stop_collect_cmd()))
            except serial.SerialException:
                pass

        self.is_collecting = False
        if hasattr(self, "btn_start"):
            self.btn_start.setText("开始采集")

    def _clear_waveform_lines(self):
        """清空波形，并把每个通道的自动量程恢复到初始值。"""
        for channel_idx, line in enumerate(self.lines):
            line.set_data([], [])
            self.channel_half_ranges[channel_idx] = self.initial_y_half_range
            self.channel_shrink_counts[channel_idx] = 0
            self.axes[channel_idx].set_ylim(
                -self.initial_y_half_range,
                self.initial_y_half_range,
            )
            value_text = self._format_scale_value(self.initial_y_half_range)
            self.y_top_labels[channel_idx].set_text(f"+{value_text}")
            self.y_bottom_labels[channel_idx].set_text(f"-{value_text}")

        self.axes[-1].set_xlim(0.0, self.display_seconds)
        self.canvas.draw_idle()

    def toggle_channel(self, state, channel_idx):
        """选中时高亮通道；取消选中时保留波形但降低亮度。"""
        enabled = bool(state)
        self.channel_states[channel_idx] = enabled

        line = self.lines[channel_idx]
        line.set_visible(True)
        if enabled:
            line.set_alpha(1.0)
            line.set_linewidth(0.90)
            line.set_zorder(20)
        else:
            line.set_alpha(0.12)
            line.set_linewidth(0.60)
            line.set_zorder(1)

        self._set_color_block_style(channel_idx, enabled)
        self.canvas.draw_idle()

    def update_show(self):
        """定时读取串口缓存、解析数据并刷新八通道波形。"""
        if self.serial_port is None or not self.serial_port.is_open:
            return

        try:
            waiting_bytes = self.serial_port.in_waiting
            if waiting_bytes <= 0:
                return

            com_data = self.serial_port.read(waiting_bytes)
        except (serial.SerialException, OSError) as exc:
            QMessageBox.critical(self, "串口错误", f"读取串口数据失败：\n{exc}")
            self._handle_serial_disconnect()
            return

        self.AdsData.parse_data(com_data)
        if not any(len(channel) > 0 for channel in self.AdsData.emg_data):
            return

        for channel_idx in range(8):
            self.update_graph(
                channel_idx,
                self.AdsData.emg_data[channel_idx],
                self.AdsData.sample_rate,
            )

        self.canvas.draw_idle()

    def update_graph(self, channel_idx, data, rate):
        """更新指定通道最近若干秒的波形。"""
        if rate <= 0 or len(data) == 0:
            self.lines[channel_idx].set_data([], [])
            return

        display_length = int(rate * self.display_seconds)
        end_index = len(data)
        start_index = max(0, end_index - display_length)

        y_data = np.asarray(data[start_index:end_index], dtype=np.float64)
        x_data = np.arange(start_index, end_index, dtype=np.float64) / float(rate)

        # 每个通道使用自己的信号幅值独立更新纵轴范围。
        # 必须在抽点之前计算，以免遗漏峰值。
        self._update_channel_y_scale(channel_idx, y_data)

        # 对显示数据进行抽点，降低高采样率下的绘图负载；原始缓存不变。
        if y_data.size > self.max_display_points:
            step = int(np.ceil(y_data.size / self.max_display_points))
            y_data = y_data[::step]
            x_data = x_data[::step]

        self.lines[channel_idx].set_data(x_data, y_data)

        # 统一更新共享时间轴，形成向左滚动的固定时间窗。
        current_time = end_index / float(rate)
        if current_time < self.display_seconds:
            x_min = 0.0
            x_max = self.display_seconds
        else:
            x_min = current_time - self.display_seconds
            x_max = current_time
        self.axes[-1].set_xlim(x_min, x_max)

    def _handle_serial_disconnect(self):
        """串口异常断开后的界面复位。"""
        self.is_collecting = False
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.btn_open.setText("打开串口")
        self.btn_start.setText("开始采集")
        self.btn_start.setEnabled(False)
        self.port_combo.setEnabled(True)
        self.btn_refresh.setEnabled(True)

    def export_csv(self):
        """将当前缓存中的八通道滤波数据导出为CSV。"""
        if not any(len(channel) > 0 for channel in self.AdsData.emg_data):
            QMessageBox.information(self, "提示", "数据缓存区为空，请先采集数据！")
            return

        now = datetime.datetime.now()
        default_name = (
            now.strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(self.AdsData.sample_rate)
            + "sps.csv"
        )

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "保存CSV文件",
            default_name,
            "CSV文件 (*.csv);;所有文件 (*)",
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"

        headers = self.channel_names

        try:
            with open(filepath, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(headers)

                max_len = max(len(channel) for channel in self.AdsData.emg_data)
                for row_index in range(max_len):
                    row = []
                    for channel_data in self.AdsData.emg_data:
                        if row_index >= len(channel_data):
                            row.append("")
                            continue

                        value = channel_data[row_index]
                        if isinstance(value, Real):
                            row.append(f"{float(value):.3f}")
                        else:
                            row.append("")
                    writer.writerow(row)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", f"CSV文件写入失败：\n{exc}")
            return

        QMessageBox.information(self, "提示", "导出成功！")

    def closeEvent(self, event):
        """关闭窗口前停止采集并释放串口。"""
        self.stop_collection()
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None
        event.accept()


def main():
    # 必须在创建QApplication之前启用高DPI支持。
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
