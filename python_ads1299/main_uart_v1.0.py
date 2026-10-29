import csv
import datetime
import sys
import serial
from serial.tools import list_ports
from numbers import Real

import numpy as np
import serial
from PyQt5.QtCore import QTimer, Qt
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
        self.y_range = 120.0
        self.max_display_points = 3000

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

        self.initUI()

    def initUI(self):
        """创建窗口、控制区和八通道波形区。"""
        self.setWindowTitle("ADS1299 八通道脑电实时显示")
        self.setGeometry(80, 80, 1600, 900)
        self.setMinimumSize(1200, 720)

        # 主窗口采用深色风格。
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #071426;
                color: #DDE7F2;
            }
            QGroupBox {
                border: 1px solid #27405D;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QComboBox, QPushButton {
                min-height: 32px;
                border: 1px solid #355574;
                border-radius: 5px;
                padding: 2px 8px;
                background-color: #10243B;
                color: #EAF2FA;
            }
            QPushButton:hover {
                background-color: #173451;
                border-color: #4E7AA3;
            }
            QPushButton:pressed {
                background-color: #0B1C30;
            }
            QPushButton:disabled, QComboBox:disabled {
                color: #687A8C;
                background-color: #0B1828;
                border-color: #23364A;
            }
            """
        )

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
            color_block.setFixedSize(18, 18)
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

    def _channel_checkbox_style(self, channel_idx):
        """生成带通道固定色的复选框样式。"""
        color = self.channel_colors[channel_idx]
        return f"""
            QCheckBox {{
                min-height: 28px;
                padding: 2px 8px;
                border-radius: 5px;
                color: #8393A5;
                background-color: #0C1B2D;
                border: 1px solid #263D55;
                font-size: 13px;
            }}
            QCheckBox:checked {{
                color: #FFFFFF;
                background-color: {color};
                border: 1px solid #FFFFFF;
                font-weight: bold;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
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
            ax.set_ylim(-self.y_range, self.y_range)
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

            # 每行中线和分隔线。
            ax.axhline(0.0, color="#112A45", linewidth=0.50, alpha=0.70)
            ax.axhline(-self.y_range, color="#152B43", linewidth=0.55, alpha=0.95)

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

            # 左侧幅值参考和通道名称。
            ax.text(
                -0.028,
                0.80,
                "+100",
                transform=ax.transAxes,
                color="#9EACBA",
                fontsize=9,
                ha="right",
                va="center",
                clip_on=False,
            )
            ax.text(
                -0.028,
                0.20,
                "-100",
                transform=ax.transAxes,
                color="#758596",
                fontsize=9,
                ha="right",
                va="center",
                clip_on=False,
            )
            ax.text(
                -0.006,
                0.50,
                self.channel_names[i],
                transform=ax.transAxes,
                color="#E6EDF5",
                fontsize=10,
                ha="right",
                va="center",
                clip_on=False,
            )

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
                    labelsize=9,
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
        """清空屏幕上的全部通道波形。"""
        for line in self.lines:
            line.set_data([], [])
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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()