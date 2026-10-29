import datetime
import sys
import time

import numpy as np
import logging
import serial
from serial.tools import list_ports
from PyQt5.QtCore import QSettings, QSignalBlocker, QStandardPaths, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ads1299 import ADS1298Data, ADS1298cmd
from alpha_model_dialog import AlphaModelDialog
from motor_imagery_dialog import MotorImageryDialog
from rhythm_analysis_dialog import RhythmAnalysisDialog
from eeg_filters import ConfigurableEEGFilter
from eye_blink import EyeBlinkDetector
from history_replay import alpha_activity_mask, read_history_csv, sample_rate_from_filename
from ring_buffer import MultiChannelRingBuffer
from link_stats import StreamLossTracker
from csv_recording import CsvExportWorker, SegmentedCsvRecorder, subject_filename_prefix

# 基本日志配置，便于调试串口与数据解析问题
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE_VID = 0x0483
DEVICE_PID = 0x5740
USB_BAUDRATE = 115200
WIFI_BAUDRATES = (1000000, 921600, 460800, 115200)
USB_READY_TIMEOUT_S = 3.0
WIFI_READY_TIMEOUT_S = 6.0
COMMAND_GAP_MS = 120
SLOW_COMMAND_GAP_MS = 350
STATUS_POLL_INTERVAL_S = 5.0
STREAM_TIMEOUT_S = 2.5


class SerialLogDialog(QDialog):
    """Small window for inspecting raw serial TX/RX traffic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("串口日志")
        self.resize(900, 520)

        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.document().setMaximumBlockCount(800)

        self.clear_button = QPushButton("清空", self)
        self.clear_button.clicked.connect(self.output.clear)
        self.save_button = QPushButton("保存日志", self)
        self.save_button.clicked.connect(self.save_log)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.clear_button)
        button_row.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.output, 1)
        layout.addLayout(button_row)

    def append_entry(self, direction, payload):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        hex_data = bytes(payload).hex(" ")
        self.output.appendPlainText(
            f"[{timestamp}] {direction} {len(payload)} bytes\n{hex_data}\n"
        )

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存串口日志", "serial.log", "Log files (*.log);;Text files (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(self.output.toPlainText())
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存日志：\n{exc}")


class SubjectDialog(QDialog):
    """采集前录入被采集者；采集时间由主窗口在真正开始时确定。"""

    def __init__(self, parent=None, subject=None):
        super().__init__(parent)
        self.setWindowTitle("被采集者信息")
        form = QFormLayout(self)
        self.kind = QComboBox(self)
        self.kind.addItem("请选择采集类型", None)
        self.kind.addItem("人的采集", "person")
        self.kind.addItem("宠物的采集", "pet")
        self.name = QLineEdit(self)
        self.age = QSpinBox(self)
        self.age.setRange(-1, 150)
        self.age.setSpecialValueText("未填写")
        self.age.setSuffix(" 岁")
        self.age.setValue(-1)
        self.weight = QDoubleSpinBox(self)
        self.weight.setRange(0.0, 500.0)
        self.weight.setSpecialValueText("未填写")
        self.weight.setDecimals(2)
        self.weight.setSuffix(" kg")
        self.time_label = QLabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self)
        form.addRow("采集类型", self.kind)
        self.name_label = QLabel("姓名", self)
        form.addRow(self.name_label, self.name)
        form.addRow("年龄", self.age)
        self.weight_label = QLabel("体重", self)
        form.addRow(self.weight_label, self.weight)
        form.addRow("采集时间", self.time_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.kind.currentIndexChanged.connect(self._update_kind)
        if subject:
            self.kind.setCurrentIndex(self.kind.findData(subject["kind"]))
            self.name.setText(subject["name"])
            self.age.setValue(subject["age"])
            if subject["kind"] == "pet":
                self.weight.setValue(subject["weight"])
        self._update_kind()

    def _update_kind(self):
        pet = self.kind.currentData() == "pet"
        self.name_label.setText("名称" if pet else "姓名")
        self.weight_label.setVisible(pet)
        self.weight.setVisible(pet)

    def _accept_validated(self):
        if self.kind.currentData() is None:
            QMessageBox.warning(self, "提示", "请选择人的采集或宠物的采集")
            return
        if not self.name.text().strip():
            QMessageBox.warning(self, "提示", "请输入姓名或名称")
            return
        if self.age.value() < 0:
            QMessageBox.warning(self, "提示", "请输入年龄")
            return
        if self.kind.currentData() == "pet" and self.weight.value() <= 0:
            QMessageBox.warning(self, "提示", "请输入宠物体重")
            return
        self.accept()

    def subject(self):
        result = {
            "kind": self.kind.currentData(),
            "name": self.name.text().strip(),
            "age": self.age.value(),
        }
        if result["kind"] == "pet":
            result["weight"] = self.weight.value()
        return result


class MainWindow(QMainWindow):
    """脑电八通道实时波形显示主窗口。"""

    recording_path = pyqtSignal(object, str)
    recording_error = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()

        # 先初始化运行状态，再创建界面和定时器。
        self.AdsData = ADS1298Data()
        self.serial_port = None
        self.serial_log_enabled = False
        self.serial_log_dialog = None
        self.alpha_model_dialog = None
        self.motor_imagery_dialog = None
        self.rhythm_analysis_dialog = None
        self.serial_log_entries = []
        self.record_settings = QSettings("EEG_PET_DESK_ZC", "Recording")
        self.record_directory = self.record_settings.value(
            "directory", QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            + "/EEG_Records", type=str,
        )
        self._recorder = None
        self._record_token = None
        self._record_cursor = 0
        self._record_error_reported = False
        self.recorded_files = []
        self.subject_info = None
        self.collection_subject = None
        self.collection_started_at = None
        self._export_worker = None
        self._export_progress = None
        self._closing_window = False
        self._record_summary_dialog = None
        self.recording_path.connect(self._on_recording_path)
        self.recording_error.connect(self._on_recording_error)
        self._link = StreamLossTracker(250)
        self._device_ready = False
        self._ready_deadline = 0.0
        self._pending_commands = []
        self._starting_stream = False
        self._next_command_at = 0.0
        self._next_status_poll = 0.0
        self._stats_started = 0.0
        self._last_eeg_host_time = None
        self._device_base = None
        self._device_drops = 0
        self._device_link_errors = 0
        self._device_report_valid = False
        self._last_rx_log_time = 0.0
        self.is_collecting = False
        self.is_impedance_testing = False
        self.is_history_mode = False
        self.history_data = None
        self.history_rate = 250
        self.history_file_path = ""
        self.alpha_occurrence_count = 0
        self.alpha_previous_activity = [False] * 8
        self.alpha_last_sample_count = 0
        self.alpha_live_active_count = None
        self.detection_band = (9.0, 13.0)
        self.eye_blink_detector = EyeBlinkDetector()
        self.eye_blink_count = 0
        self.impedance_labels = []
        self.gravity_labels = []
        self.channel_states = [True] * 8

        # 实时滤波状态。
        # EEGFilters内部保存IIR延迟状态，因此每次只送入新增采样点。
        self.eeg_filter = ConfigurableEEGFilter()
        self.filter_enabled = False
        self.filter_sample_rate = None
        self.filtered_source_length = 0
        self.filtered_emg_data = [[] for _ in range(8)]

        # 波形显示参数。
        self.display_seconds = 10.0
        self.max_display_points = 3000
        self.plot_buffer_rate = None
        self.raw_plot_buffer = None
        self.display_plot_buffer = None
        self.display_plot_source = None
        self.spectrum_seconds = 4.0
        self.spectrum_max_frequency = 100.0
        self.user_spectrum_xlim = None
        self.user_spectrum_ylim = None

        # 与参考平台一致，波形默认以每通道 +/-100 uV 显示。
        # 滚轮可手动调整幅值，双击恢复初始范围。
        self.initial_y_half_range = 100.0
        # 通道名称及固定颜色。
        self.channel_names = ["F5", "F3", "FZ", "F4", "F6", "AF8", "FPZ", "AF7"]
        self.channel_name_history = [tuple(self.channel_names)]
        self.channel_colors = [
            "#2F80FF",  # F5 蓝色
            "#FF7A18",  # F3 橙色
            "#F04A9B",  # FZ 粉色
            "#20C878",  # F4 绿色
            "#8D6BEB",  # F6 紫色
            "#32B7E9",  # AF8 青蓝色
            "#E3A600",  # FPZ 黄色
            "#12B8B5",  # AF7 青色
        ]

        self.lines = []
        self.highlight_lines = []
        self.spectrum_lines = []
        self.spectrum_ax = None
        self.spectrum_peak_marker = None
        self.spectrum_peak_annotation = None
        self.channel_buttons = []
        self.channel_color_blocks = []
        self.y_top_labels = []
        self.y_bottom_labels = []
        self.channel_name_labels = []
        self.channel_panel_expanded = True

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
        self.setWindowTitle("智宠未来脑电实时显示")
        self.setGeometry(80, 80, 1600, 900)
        self.setMinimumSize(1200, 720)

        # 根据当前屏幕分辨率、DPI和窗口大小计算字体缩放比例。
        self._update_scaled_metrics()
        self._apply_responsive_style()

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ---------------- 左侧控制区 ----------------
        left_vbox = QVBoxLayout()
        left_vbox.setAlignment(Qt.AlignTop)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(6)

        port_row = QHBoxLayout()
        port_label = QLabel("选择串口")
        port_label.setStyleSheet("color: #E8F0FB; font-size: 14px; font-weight: 500;")
        self.port_combo = QComboBox(self)
        self.port_combo.setMinimumHeight(34)
        self.port_combo.setStyleSheet("QComboBox { border-radius: 8px; padding-left: 10px; }")
        port_row.addWidget(port_label)
        port_row.addWidget(self.port_combo, 1)

        self.wifi_check = QCheckBox("使用 WiFi 模块", self)
        self.wifi_check.setToolTip("通过 WiFi 模块虚拟串口透传，采样率固定 250 SPS")
        baud_row = QHBoxLayout()
        baud_label = QLabel("波特率", self)
        self.baud_combo = QComboBox(self)
        self.baud_combo.setMinimumHeight(34)
        for rate in (USB_BAUDRATE, 460800, 921600, 1000000):
            self.baud_combo.addItem(f"{rate:,} bps", rate)
        self.baud_combo.setEnabled(False)
        baud_row.addWidget(baud_label)
        baud_row.addWidget(self.baud_combo, 1)
        self.wifi_check.toggled.connect(self._on_wifi_toggled)

        self.btn_refresh = QPushButton("刷新串口", self)
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_refresh.setMinimumHeight(self.control_height)

        self.btn_open = QPushButton("打开串口", self)
        self.btn_open.clicked.connect(self.open_serial)
        self.btn_open.setMinimumHeight(self.control_height)

        self.btn_serial_log = QPushButton("串口日志", self)
        self.btn_serial_log.setCheckable(True)
        self.btn_serial_log.setToolTip("显示或隐藏原始串口收发日志")
        self.btn_serial_log.clicked.connect(self.toggle_serial_log)
        self.btn_serial_log.setMinimumHeight(self.control_height)

        self.btn_start = QPushButton("▶ 开始采集", self)
        self.btn_start.clicked.connect(self.start_collect)
        self.btn_start.setEnabled(False)
        self.btn_start.setMinimumHeight(self.control_height)

        self.btn_stop = QPushButton("■ 停止采集", self)
        self.btn_stop.clicked.connect(self.stop_collection)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(self.control_height)

        self.btn_subject = QPushButton("录入被采集者信息", self)
        self.btn_subject.clicked.connect(self.edit_subject)
        self.btn_subject.setMinimumHeight(self.control_height)

        self.btn_record = QPushButton("开始保存数据", self)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        self.btn_record.setMinimumHeight(self.control_height)

        self.btn_filter = QPushButton("带通滤波", self)
        self.btn_filter.setObjectName("filterButton")
        self.btn_filter.setCheckable(True)
        self.btn_filter.setChecked(False)
        self.btn_filter.setToolTip("按下后按下方截止频率过滤 EEG 数据")
        self.btn_filter.toggled.connect(self.toggle_filter)

        self.low_cutoff_label = QLabel("下限截止频率", self)
        self.low_cutoff = QDoubleSpinBox(self)
        self.low_cutoff.setRange(0.1, 119.9)
        self.low_cutoff.setDecimals(2)
        self.low_cutoff.setSingleStep(0.1)
        self.low_cutoff.setSuffix(" Hz")
        self.low_cutoff.setValue(0.1)

        self.high_cutoff_label = QLabel("上限截止频率", self)
        self.high_cutoff = QDoubleSpinBox(self)
        self.high_cutoff.setRange(0.2, 120.0)
        self.high_cutoff.setDecimals(2)
        self.high_cutoff.setSingleStep(1.0)
        self.high_cutoff.setSuffix(" Hz")
        self.high_cutoff.setValue(100.0)

        self.notch_checkbox = QCheckBox("50 Hz 陷波", self)
        self.notch_checkbox.setToolTip("抑制 50 Hz 工频干扰")
        self.dc_checkbox = QCheckBox("去直流 (0.1 Hz)", self)
        self.dc_checkbox.setToolTip("移除 EEG 基线漂移；启用带通时已包含去直流")
        self.low_cutoff.valueChanged.connect(self._filter_settings_changed)
        self.high_cutoff.valueChanged.connect(self._filter_settings_changed)
        self.notch_checkbox.toggled.connect(self._filter_settings_changed)
        self.dc_checkbox.toggled.connect(self._filter_settings_changed)

        channel_group = QGroupBox()
        channel_group.setObjectName("channelGroup")
        channel_group.setStyleSheet(
            "QGroupBox#channelGroup { border: 1px solid #2B4462; border-radius: 9px; background-color: #0B1D2E; margin-top: 4px; }"
            "QGroupBox#channelGroup::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.setContentsMargins(8, 10, 8, 8)
        channel_layout.setSpacing(5)

        channel_title_row = QHBoxLayout()
        channel_title = QLabel("通道选择")
        channel_title.setStyleSheet("color: #E8F0FB; font-size: 13px; font-weight: bold;")
        self.channel_toggle_btn = QPushButton("︿")
        self.channel_toggle_btn.setFixedSize(28, 28)
        self.channel_toggle_btn.setObjectName("channelToggleButton")
        self.channel_toggle_btn.setCheckable(True)
        self.channel_toggle_btn.setChecked(True)
        self.channel_toggle_btn.clicked.connect(self.toggle_channel_panel)
        self.channel_toggle_btn.setStyleSheet(
            "QPushButton#channelToggleButton { border: 1px solid #355574; border-radius: 6px; background: #10243B; color: #EAF2FA; font-size: 18px; font-weight: bold; }"
            "QPushButton#channelToggleButton:hover { background: #173451; border-color: #4E7AA3; }"
        )
        channel_title_row.addWidget(channel_title)
        channel_title_row.addStretch(1)
        channel_title_row.addWidget(self.channel_toggle_btn)
        channel_layout.addLayout(channel_title_row)

        self.channel_panel = QWidget()
        self.channel_panel.setVisible(True)
        self.channel_panel.setMaximumHeight(9999)
        self.channel_panel.setMinimumHeight(0)
        self.channel_panel.setObjectName("channelPanel")
        self.channel_panel_layout = QVBoxLayout(self.channel_panel)
        self.channel_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_panel_layout.setSpacing(4)

        for i in range(8):
            row = QHBoxLayout()
            row.setContentsMargins(6, 4, 6, 4)
            row.setSpacing(8)

            color_block = QLabel(self)
            color_block.setFixedSize(self.color_block_size + 2, self.color_block_size + 2)
            self._set_color_block_style(i, enabled=True, widget=color_block)

            checkbox = QCheckBox(self.channel_names[i], self)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(self._channel_checkbox_style(i))
            checkbox.toggled.connect(
                lambda state, idx=i: self.toggle_channel(state, idx)
            )

            row.addWidget(color_block)
            row.addWidget(checkbox, 1)

            row_widget = QWidget(self)
            row_widget.setLayout(row)
            row_widget.setStyleSheet(
                f"background-color: {self.channel_colors[i]}; border: 1px solid rgba(255,255,255,0.22); border-radius: 4px;"
            )
            row_widget.setMinimumHeight(max(28, round(30 * self.ui_scale)))
            self.channel_panel_layout.addWidget(row_widget)

            self.channel_color_blocks.append(color_block)
            self.channel_buttons.append(checkbox)

        channel_layout.addWidget(self.channel_panel)

        left_vbox.addLayout(port_row)
        left_vbox.addWidget(self.wifi_check)
        left_vbox.addLayout(baud_row)
        left_vbox.addWidget(self.btn_refresh)
        left_vbox.addWidget(self.btn_open)
        left_vbox.addWidget(self.btn_serial_log)
        left_vbox.addWidget(self.btn_start)
        left_vbox.addWidget(self.btn_stop)
        left_vbox.addWidget(self.btn_filter)
        left_vbox.addWidget(self.low_cutoff_label)
        left_vbox.addWidget(self.low_cutoff)
        left_vbox.addWidget(self.high_cutoff_label)
        left_vbox.addWidget(self.high_cutoff)
        left_vbox.addWidget(self.notch_checkbox)
        left_vbox.addWidget(self.dc_checkbox)
        left_vbox.addWidget(channel_group)
        left_vbox.addStretch(1)

        # ---------------- 右侧波形区 ----------------
        right_vbox = QVBoxLayout()
        right_vbox.setContentsMargins(0, 0, 0, 0)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        self.btn_impedance = QPushButton("阻抗检测", self)
        self.btn_impedance.setCheckable(True)
        self.btn_impedance.setEnabled(False)
        self.btn_impedance.setToolTip("启动或停止 8 通道阻抗检测")
        self.btn_impedance.clicked.connect(self.toggle_impedance_test)
        self.btn_impedance.setMinimumHeight(self.control_height)
        top_bar.addWidget(self.btn_impedance)
        self.impedance_toggle_btn = QPushButton("▼", self)
        self.impedance_toggle_btn.setCheckable(True)
        self.impedance_toggle_btn.setChecked(True)
        self.impedance_toggle_btn.setFixedSize(max(28, self.control_height - 6), max(28, self.control_height - 6))
        self.impedance_toggle_btn.setToolTip("展开或收起阻抗结果")
        self.impedance_toggle_btn.toggled.connect(self.toggle_impedance_panel)
        top_bar.addWidget(self.impedance_toggle_btn)
        self.btn_export = QPushButton("导出 CSV", self)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setToolTip("保存当前八通道数据")
        top_bar.addWidget(self.btn_export)
        self.btn_history = QPushButton("历史回访", self)
        self.btn_history.clicked.connect(self.open_history)
        self.btn_history.setToolTip("打开以前导出的 EEG CSV 文件")
        top_bar.addWidget(self.btn_history)
        self.btn_live = QPushButton("返回实时", self)
        self.btn_live.clicked.connect(self.leave_history)
        self.btn_live.setVisible(False)
        top_bar.addWidget(self.btn_live)
        top_bar.addStretch(1)
        self.eye_blink_status = QLabel("睁闭眼估计 3分累计0次", self)
        self.eye_blink_status.setStyleSheet("color: #8CE4D0; font-weight: bold;")
        self.eye_blink_status.setToolTip(
            "根据前额 EEG 估计完整眨眼（闭眼后再睁眼），每次计 1 次；首次出现后 3 分钟清零。"
            "此数值不是设备直接上报的睁眼或闭眼状态。"
        )
        top_bar.addWidget(self.eye_blink_status)
        self.waveform_mode_title = QLabel("EEG 实时波形", self)
        top_bar.addWidget(self.waveform_mode_title)
        right_vbox.addLayout(top_bar)

        subject_row = QHBoxLayout()
        subject_row.addWidget(self.btn_subject)
        self.subject_status_label = QLabel("未录入被采集者信息", self)
        subject_row.addWidget(self.subject_status_label)
        subject_row.addStretch(1)
        subject_row.addWidget(self.btn_record)
        right_vbox.addLayout(subject_row)

        record_row = QHBoxLayout()
        self.btn_record_directory = QPushButton("选择保存目录", self)
        self.btn_record_directory.clicked.connect(self.select_record_directory)
        self.btn_record_directory.setToolTip(self.record_directory)
        record_row.addWidget(self.btn_record_directory)
        record_row.addWidget(QLabel("每段时长：", self))
        self.record_seconds = QSpinBox(self)
        self.record_seconds.setRange(1, 86400)
        self.record_seconds.setSuffix(" 秒")
        self.record_seconds.setValue(self.record_settings.value("segment_seconds", 300, type=int))
        record_row.addWidget(self.record_seconds)
        record_row.addWidget(QLabel("每段行数上限：", self))
        self.record_max_rows = QSpinBox(self)
        self.record_max_rows.setRange(0, 100000000)
        self.record_max_rows.setSpecialValueText("不限行数")
        self.record_max_rows.setValue(self.record_settings.value("max_rows", 0, type=int))
        self.record_max_rows.setToolTip("时长与行数任一达到上限即切分；0 表示只按时长切分")
        record_row.addWidget(self.record_max_rows)
        self.record_seconds.valueChanged.connect(self._save_record_limits)
        self.record_max_rows.valueChanged.connect(self._save_record_limits)
        record_row.addStretch(1)
        right_vbox.addLayout(record_row)

        band_row = QHBoxLayout()
        band_row.setSpacing(6)
        band_row.addWidget(QLabel("波形检测频段", self))
        self.band_low = QDoubleSpinBox(self)
        self.band_high = QDoubleSpinBox(self)
        for spin in (self.band_low, self.band_high):
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" Hz")
            spin.setFixedWidth(94)
            spin.setKeyboardTracking(False)
        self.band_low.setRange(1.0, 39.5)
        self.band_high.setRange(1.5, 40.0)
        self.band_low.setValue(9.0)
        self.band_high.setValue(13.0)
        self.band_low.valueChanged.connect(self._on_detection_band_changed)
        self.band_high.valueChanged.connect(self._on_detection_band_changed)
        band_row.addWidget(self.band_low)
        band_row.addWidget(QLabel("至", self))
        band_row.addWidget(self.band_high)
        self.alpha_status = QLabel("9–13 Hz -- · 3分累计0次", self)
        self.alpha_status.setStyleSheet("color: #EEC75A; font-weight: bold;")
        self.alpha_status.setToolTip(
            "黄色波形表示该时间段以所选频段为主；每个通道从未命中到命中计一次，首次出现后 3 分钟清零"
        )
        band_row.addWidget(self.alpha_status)
        band_row.addStretch(1)
        right_vbox.addLayout(band_row)

        impedance_group = QGroupBox(self)
        self.impedance_group = impedance_group
        impedance_layout = QHBoxLayout(impedance_group)
        impedance_layout.setContentsMargins(8, 4, 8, 4)
        impedance_layout.setSpacing(8)
        for index, name in enumerate(self.channel_names):
            label = QLabel(f"{name}\n-- kΩ", impedance_group)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(40)
            label.setObjectName("impedanceValue")
            label.setWordWrap(True)
            impedance_layout.addWidget(label, 1)
            self.impedance_labels.append(label)
        right_vbox.addWidget(impedance_group)

        main_title = QLabel("智宠未来脑电实时显示")
        main_title.setStyleSheet(
            "color: #E8F0FB; font-size: 22px; font-weight: bold; "
            "padding: 0 0 4px 0;"
        )
        main_title.setAlignment(Qt.AlignLeft)
        model_title_row = QHBoxLayout()
        model_title_row.addWidget(main_title)
        model_title_row.addStretch(1)
        self.btn_alpha_model = QPushButton("脑电控制", self)
        self.btn_alpha_model.setToolTip("用校准后的 9–13 Hz 功率控制小球位置")
        self.btn_alpha_model.clicked.connect(self.open_alpha_model)
        model_title_row.addWidget(self.btn_alpha_model)
        self.btn_motor_imagery = QPushButton("运动想象", self)
        self.btn_motor_imagery.setToolTip("使用 C3、C4 通道训练左右手运动想象，再控制小球")
        self.btn_motor_imagery.clicked.connect(self.open_motor_imagery)
        model_title_row.addWidget(self.btn_motor_imagery)
        self.btn_rhythm_analysis = QPushButton("节律分析", self)
        self.btn_rhythm_analysis.clicked.connect(self.open_rhythm_analysis)
        model_title_row.addWidget(self.btn_rhythm_analysis)
        right_vbox.addLayout(model_title_row)

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
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)

        self.history_controls = QWidget(self)
        history_row = QHBoxLayout(self.history_controls)
        history_row.setContentsMargins(0, 0, 0, 0)
        self.btn_history_play = QPushButton("▶", self.history_controls)
        self.btn_history_play.setFixedWidth(44)
        self.btn_history_play.setToolTip("播放或暂停历史数据")
        self.btn_history_play.clicked.connect(self.toggle_history_playback)
        self.history_slider = QSlider(Qt.Horizontal, self.history_controls)
        self.history_slider.setTracking(False)
        self.history_slider.valueChanged.connect(self._render_history)
        self.history_time_label = QLabel("00:00 / 00:00", self.history_controls)
        self.history_time_label.setMinimumWidth(190)
        history_row.addWidget(self.btn_history_play)
        history_row.addWidget(self.history_slider, 1)
        history_row.addWidget(self.history_time_label)
        right_vbox.addWidget(self.history_controls)
        self.history_controls.setVisible(False)

        self.history_timer = QTimer(self)
        self.history_timer.timeout.connect(self._advance_history)

        self.alpha_reset_timer = QTimer(self)
        self.alpha_reset_timer.setSingleShot(True)
        self.alpha_reset_timer.setTimerType(Qt.PreciseTimer)
        self.alpha_reset_timer.timeout.connect(self._reset_alpha_occurrences)

        self.eye_blink_reset_timer = QTimer(self)
        self.eye_blink_reset_timer.setSingleShot(True)
        self.eye_blink_reset_timer.setTimerType(Qt.PreciseTimer)
        self.eye_blink_reset_timer.timeout.connect(self._reset_eye_blink_count)

        gravity_group = QGroupBox("重力感应", self)
        gravity_layout = QHBoxLayout(gravity_group)
        gravity_layout.setContentsMargins(8, 4, 8, 4)
        gravity_layout.setSpacing(12)
        for axis in ("X", "Y", "Z", "|G|"):
            label = QLabel(f"加速度 {axis}: --", gravity_group)
            label.setMinimumHeight(28)
            gravity_layout.addWidget(label, 1)
            self.gravity_labels.append(label)
        self.gravity_state_label = QLabel("状态: --", gravity_group)
        gravity_layout.addWidget(self.gravity_state_label, 1)
        right_vbox.addWidget(gravity_group)

        self.spectrum_title = QLabel("EEG 频谱（实时） 频率（Hz）", self)
        self.spectrum_title.setStyleSheet(
            "color: #C4CFDB; font-size: 14px; font-weight: bold; padding: 2px 0 0 0;"
        )
        right_vbox.addWidget(self.spectrum_title)

        self.spectrum_fig, self.spectrum_ax = plt.subplots(
            1,
            1,
            figsize=(16, 2.2),
        )
        self.spectrum_fig.patch.set_facecolor("#071426")
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_canvas.setStyleSheet("background-color: #071426; border: none;")
        self.spectrum_canvas.setMinimumHeight(145)
        self.spectrum_canvas.setMaximumHeight(160)
        self._init_spectrum_axes()
        right_vbox.addWidget(self.spectrum_canvas, 0)
        self.spectrum_canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.spectrum_canvas.mpl_connect("button_press_event", self._on_plot_button_press)

        root_layout.addLayout(left_vbox, 0)
        root_layout.addLayout(right_vbox, 1)

        central_widget = QWidget(self)
        central_widget.setLayout(root_layout)
        self.setCentralWidget(central_widget)

        # 每100 ms读取串口并刷新波形。
        self.update_show_timer = QTimer(self)
        self.update_show_timer.timeout.connect(self.update_show)
        self.update_show_timer.start(100)

        self.command_timer = QTimer(self)
        self.command_timer.setSingleShot(True)
        self.command_timer.timeout.connect(self._run_next_command)
        self.ready_timer = QTimer(self)
        self.ready_timer.setInterval(200)
        self.ready_timer.timeout.connect(self._check_device_ready)
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(1000)
        self.stats_timer.timeout.connect(self._update_link_display)
        self.stats_timer.start()
        self.link_status_label = QLabel("链路：未连接", self)
        self.record_status_label = QLabel("自动记录：待采集", self)
        self.record_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.statusBar().addPermanentWidget(self.record_status_label, 1)
        self.statusBar().addPermanentWidget(self.link_status_label)

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
        self.control_height = max(34, round(36 * self.ui_scale))
        self.checkbox_height = max(30, round(32 * self.ui_scale))
        self.checkbox_indicator_size = max(16, round(17 * self.ui_scale))
        self.color_block_size = max(19, round(20 * self.ui_scale))

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
            QGroupBox#channelGroup {{
                border: 1px solid #2A425E;
                border-radius: 10px;
                background-color: #0B1D2E;
                margin-top: 8px;
            }}
            QGroupBox#channelGroup::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{
                font-size: {self.ui_font_px}px;
            }}
            QLabel#impedanceValue {{
                color: #DDE7F2;
                background-color: #12324A;
                border: 1px solid #2E617E;
                border-radius: 6px;
                padding: 4px;
                font-weight: 600;
            }}
            QGroupBox {{
                border: 1px solid #2A425E;
                border-radius: {max(8, round(10 * self.ui_scale))}px;
                margin-top: {max(8, round(10 * self.ui_scale))}px;
                padding-top: {max(6, round(8 * self.ui_scale))}px;
                font-size: {max(14, self.ui_font_px - 1)}px;
                font-weight: bold;
                background-color: #0B1D2E;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {horizontal_padding}px;
                padding: 0 {max(4, round(5 * self.ui_scale))}px;
            }}
            QComboBox, QPushButton {{
                min-height: {self.control_height}px;
                border: 1px solid #355574;
                border-radius: {max(8, round(9 * self.ui_scale))}px;
                padding: 2px {max(6, round(7 * self.ui_scale))}px;
                background-color: #10243B;
                color: #EAF2FA;
                font-size: {max(12, self.ui_font_px - 2)}px;
                font-weight: 500;
            }}
            QPushButton#channelToggleButton {{
                border: 1px solid #355574;
                border-radius: 6px;
                background: #10243B;
                color: #EAF2FA;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#channelToggleButton:hover {{
                background: #173451;
                border-color: #4E7AA3;
            }}
            QComboBox QAbstractItemView {{
                background-color: #10243B;
                color: #EAF2FA;
                selection-background-color: #24517A;
                font-size: {self.ui_font_px}px;
                min-height: 42px;
            }}
            QPushButton:hover {{
                background-color: #173451;
                border-color: #4E7AA3;
            }}
            QPushButton:pressed {{
                background-color: #0B1C30;
            }}
            QPushButton#filterButton:checked {{
                background-color: #16784A;
                border: 1px solid #63E6A5;
                color: #FFFFFF;
                font-weight: bold;
            }}
            QPushButton#filterButton:checked:hover {{
                background-color: #1B8D58;
            }}
            QDoubleSpinBox {{
                min-height: {max(28, self.control_height - 4)}px;
                border: 1px solid #355574;
                border-radius: 4px;
                padding: 1px 4px;
                background-color: #071426;
                color: #EAF2FA;
            }}
            QCheckBox {{
                min-height: {max(25, self.control_height - 8)}px;
                spacing: 7px;
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
            left_margin = self._waveform_left_margin()
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
            self.fig.subplots_adjust(left=self._waveform_left_margin())
            self.canvas.draw_idle()
            return
        self._apply_responsive_style()

    def _waveform_left_margin(self):
        base = float(np.clip(0.085 + 0.025 * (self.ui_scale - 1.0), 0.085, 0.135))
        font = QFont("Microsoft YaHei")
        font.setPixelSize(max(12, round(self.plot_channel_font_px * 1.5)))
        width = max(QFontMetrics(font).horizontalAdvance(name) for name in self.channel_names)
        return min(0.32, max(base, (width + 16) / max(self.canvas.width(), 1)))

    def resizeEvent(self, event):
        """窗口缩放时触发响应式字体更新。"""
        super().resizeEvent(event)
        if hasattr(self, "font_resize_timer"):
            self.font_resize_timer.start(160)

    def edit_channel_name(self, channel_idx):
        current = self.channel_names[channel_idx]
        value, confirmed = QInputDialog.getText(
            self, "修改通道名称", f"通道 {channel_idx + 1} 名称", text=current,
        )
        if not confirmed:
            return
        name = value.strip()
        if not name or len(name) > 12:
            QMessageBox.warning(self, "名称无效", "通道名称须为 1 到 12 个字符。")
            return
        if any(name.casefold() == other.casefold() for index, other in enumerate(self.channel_names)
               if index != channel_idx):
            QMessageBox.warning(self, "名称重复", "通道名称不能重复。")
            return
        if name == current:
            return

        self.channel_name_history.append(tuple(self.channel_names))
        self.channel_names[channel_idx] = name
        self.channel_buttons[channel_idx].setText(name)
        self.channel_buttons[channel_idx].setToolTip(name)
        self.channel_name_labels[channel_idx].set_text(name)
        impedance_label = self.impedance_labels[channel_idx]
        impedance_label.setText(f"{name}\n{impedance_label.text().split(chr(10))[-1]}")
        impedance_label.setToolTip(name)
        self.spectrum_lines[channel_idx].set_label(name)
        self.spectrum_ax.get_legend().get_texts()[channel_idx].set_text(name)
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.channel_checks[channel_idx].setText(name)
        if self.motor_imagery_dialog is not None:
            for combo in (self.motor_imagery_dialog.c3_combo, self.motor_imagery_dialog.c4_combo):
                combo.setItemText(channel_idx + 1, f"CH{channel_idx + 1}（主界面 {name}）")
        if self.rhythm_analysis_dialog is not None:
            self.rhythm_analysis_dialog.rename_channel(channel_idx, name)
        self.fig.subplots_adjust(left=self._waveform_left_margin())
        self.canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def toggle_impedance_panel(self, expanded):
        """Show or hide the impedance result panel."""
        self.impedance_group.setVisible(bool(expanded))
        self.impedance_toggle_btn.setText("▼" if expanded else "▶")

    def toggle_channel_panel(self, checked=None):
        """切换通道选择面板的展开/收起状态。"""
        if checked is None:
            checked = self.channel_toggle_btn.isChecked()

        self.channel_panel_expanded = checked
        self.channel_panel.setVisible(checked)
        self.channel_panel.setMaximumHeight(9999 if checked else 0)
        self.channel_panel.setMinimumHeight(0 if not checked else 0)
        self.channel_toggle_btn.setText("︿" if checked else "﹀")
        self.channel_toggle_btn.setToolTip("展开通道选择" if not checked else "收起通道选择")

    def _channel_checkbox_style(self, channel_idx):
        """生成带通道固定色的复选框样式。"""
        color = self.channel_colors[channel_idx]
        return f"""
            QCheckBox {{
                min-height: {self.checkbox_height}px;
                padding: 2px {max(8, round(9 * self.ui_scale))}px;
                border-radius: 5px;
                color: #FFFFFF;
                background: transparent;
                border: none;
                font-size: {max(15, self.ui_font_px - 1)}px;
                font-weight: bold;
            }}
            QCheckBox:checked {{
                color: #FFFFFF;
                font-weight: bold;
            }}
            QCheckBox::indicator {{
                width: {self.checkbox_indicator_size}px;
                height: {self.checkbox_indicator_size}px;
                border: 2px solid rgba(255,255,255,0.9);
                border-radius: 4px;
                background: rgba(255,255,255,0.14);
            }}
            QCheckBox::indicator:checked {{
                background: rgba(255,255,255,0.96);
                border: 2px solid rgba(255,255,255,0.9);
            }}
            QCheckBox::indicator:checked:hover {{
                background: rgba(255,255,255,1.0);
            }}
        """

    def _set_color_block_style(self, channel_idx, enabled, widget=None):
        """设置通道固定色块的高亮或弱化状态。"""
        target = widget if widget is not None else self.channel_color_blocks[channel_idx]
        color = self.channel_colors[channel_idx]
        if enabled:
            target.setStyleSheet(
                f"background-color: {color}; border: 2px solid rgba(255,255,255,0.85); border-radius: 4px;"
            )
        else:
            target.setStyleSheet(
                f"background-color: {color}; border: 1px solid rgba(255,255,255,0.26); border-radius: 4px;"
            )

    def _init_spectrum_axes(self):
        """Initialize the compact real-time EEG power spectrum plot."""
        ax = self.spectrum_ax
        ax.set_facecolor("#0A1729")
        ax.set_xlim(0.2, self.spectrum_max_frequency)
        ax.set_ylim(-100.0, 25.0)
        ax.set_ylabel("功率 (dB)", color="#AAB8C7", fontsize=self.plot_scale_font_px)
        ax.set_xlabel("", color="#C4CFDB", fontsize=self.plot_scale_font_px)
        ax.tick_params(axis="both", colors="#AAB8C7", labelsize=self.plot_scale_font_px, length=0)
        ax.xaxis.set_major_locator(MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(MultipleLocator(5.0))
        ax.grid(True, which="major", axis="both", color="#183452", linewidth=0.55, alpha=0.8)
        ax.grid(True, which="minor", axis="x", color="#102943", linewidth=0.35, alpha=0.55)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#50647A")
        ax.spines["bottom"].set_color("#50647A")
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)

        for channel_idx, name in enumerate(self.channel_names):
            line, = ax.plot(
                [],
                [],
                color=self.channel_colors[channel_idx],
                linewidth=0.9,
                label=name,
                antialiased=True,
            )
            self.spectrum_lines.append(line)

        legend = ax.legend(
            loc="upper right",
            ncol=4,
            frameon=False,
            fontsize=max(7, self.plot_scale_font_px - 2),
            labelcolor="#C4CFDB",
            handlelength=1.4,
            columnspacing=0.8,
            borderaxespad=0.2,
        )
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#C4CFDB")
        self.spectrum_peak_marker, = ax.plot(
            [], [], linestyle="none", marker="o", markersize=4.5,
            markeredgecolor="#EAF2FA", markeredgewidth=0.7,
            label="_nolegend_", zorder=7,
        )
        self.spectrum_peak_marker.set_visible(False)
        self.spectrum_peak_annotation = ax.annotate(
            "", xy=(0, 0), xytext=(7, -9), textcoords="offset points",
            ha="left", va="top", fontsize=max(8, self.plot_scale_font_px - 1),
            color="#EAF2FA", zorder=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#0B1D2E",
                  "edgecolor": "#50647A", "alpha": 0.96},
        )
        self.spectrum_peak_annotation.set_visible(False)
        self.spectrum_fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.28)

    def _update_spectrum_peak_annotation(self):
        """Mark the highest displayed power among the enabled EEG channels."""
        peak = None
        x_min, x_max = sorted(self.spectrum_ax.get_xlim())
        for line in self.spectrum_lines:
            if not line.get_visible():
                continue
            frequencies = np.asarray(line.get_xdata(), dtype=np.float64)
            power_db = np.asarray(line.get_ydata(), dtype=np.float64)
            if frequencies.size == 0 or frequencies.size != power_db.size:
                continue
            in_view = (
                np.isfinite(frequencies) & np.isfinite(power_db)
                & (frequencies >= x_min) & (frequencies <= x_max)
            )
            if not np.any(in_view):
                continue
            visible_indices = np.flatnonzero(in_view)
            index = visible_indices[np.argmax(power_db[in_view])]
            if peak is None or power_db[index] > peak[1]:
                peak = (float(frequencies[index]), float(power_db[index]), line.get_color())

        if peak is None:
            self.spectrum_peak_marker.set_data([], [])
            self.spectrum_peak_marker.set_visible(False)
            self.spectrum_peak_annotation.set_visible(False)
            return

        frequency, power, color = peak
        self.spectrum_peak_marker.set_data([frequency], [power])
        self.spectrum_peak_marker.set_markerfacecolor(color)
        self.spectrum_peak_marker.set_visible(True)
        annotation = self.spectrum_peak_annotation
        annotation.xy = (frequency, power)
        annotation.set_text(f"{frequency:.1f} Hz  {power:.1f} dB")
        if frequency > x_min + 0.6 * (x_max - x_min):
            annotation.set_position((-7, -9))
            annotation.set_ha("right")
        else:
            annotation.set_position((7, -9))
            annotation.set_ha("left")
        annotation.get_bbox_patch().set_edgecolor(color)
        annotation.set_visible(True)

    def _update_spectrum(self, data_source, rate):
        """Calculate and draw the latest 0.2-100 Hz spectrum for all channels."""
        if self.spectrum_ax is None or rate <= 0:
            return

        sample_count = max(16, int(round(rate * self.spectrum_seconds)))
        all_power = []
        frequency_axis = None
        for channel_idx, line in enumerate(self.spectrum_lines):
            channel_data = data_source[channel_idx] if channel_idx < len(data_source) else []
            values = np.asarray(channel_data[-sample_count:], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size < max(16, rate // 2):
                line.set_data([], [])
                continue

            values = values - np.mean(values)
            window = np.hanning(values.size)
            fft_size = max(256, 1 << int(np.ceil(np.log2(values.size))))
            spectrum = np.fft.rfft(values * window, n=fft_size)
            power = (np.abs(spectrum) ** 2) / max(np.sum(window ** 2), 1.0)
            frequencies = np.fft.rfftfreq(fft_size, 1.0 / float(rate))
            visible = (frequencies >= 0.2) & (frequencies <= self.spectrum_max_frequency)
            frequencies = frequencies[visible]
            power_db = 10.0 * np.log10(np.maximum(power[visible], 1e-12))
            line.set_data(frequencies, power_db)
            line.set_visible(self.channel_states[channel_idx])
            frequency_axis = frequencies
            all_power.append(power_db)

        if frequency_axis is not None and all_power:
            finite_power = np.concatenate(all_power)
            peak = float(np.max(finite_power))
            floor = float(np.percentile(finite_power, 5.0))
            upper = max(20.0, np.ceil(peak + 6.0))
            lower = min(-100.0, np.floor(floor - 8.0))
            if upper - lower < 55.0:
                lower = upper - 55.0
            if self.user_spectrum_ylim is None:
                self.spectrum_ax.set_ylim(lower, upper)
        if self.user_spectrum_xlim is not None:
            self.spectrum_ax.set_xlim(*self.user_spectrum_xlim)
        self._update_spectrum_peak_annotation()

    @staticmethod
    def _zoom_limits(limits, center, factor):
        low, high = map(float, limits)
        if not np.isfinite(center):
            center = (low + high) / 2.0
        new_low = center - (center - low) * factor
        new_high = center + (high - center) * factor
        if new_high - new_low < 1e-6:
            midpoint = (new_low + new_high) / 2.0
            new_low, new_high = midpoint - 5e-7, midpoint + 5e-7
        return new_low, new_high

    def _on_plot_scroll(self, event):
        """只缩放纵轴，保持实时波形的时间窗口连续滚动。"""
        if event.inaxes is None:
            return
        step = float(getattr(event, "step", 0) or 0)
        if step == 0:
            return
        factor = 0.8 if step > 0 else 1.25

        waveform_index = next(
            (index for index, axis in enumerate(self.axes) if event.inaxes is axis),
            None,
        )
        if waveform_index is not None:
            for index, axis in enumerate(self.axes):
                current_ylim = axis.get_ylim()
                half_range = max(abs(current_ylim[0]), abs(current_ylim[1]))
                half_range = float(np.clip(half_range * factor, 0.1, 1e9))
                y_limits = (-half_range, half_range)
                axis.set_ylim(*y_limits)
                self.y_top_labels[index].set_text(
                    f"+{self._format_scale_value(half_range)}"
                )
                self.y_bottom_labels[index].set_text(
                    f"-{self._format_scale_value(half_range)}"
                )
            self.canvas.draw_idle()
            return

        if event.inaxes is self.spectrum_ax:
            current_ylim = self.spectrum_ax.get_ylim()
            y_center = event.ydata if event.ydata is not None else sum(current_ylim) / 2.0
            y_limits = self._zoom_limits(current_ylim, float(y_center), factor)
            self.user_spectrum_ylim = y_limits
            self.spectrum_ax.set_ylim(*y_limits)
            self.spectrum_canvas.draw_idle()

    def _on_plot_button_press(self, event):
        """双击波形恢复初始幅值，双击频谱恢复其自动纵轴。"""
        if event.button != 1 or not event.dblclick:
            return
        if getattr(event, "canvas", None) is self.canvas and hasattr(event, "x") and hasattr(event, "y"):
            for index, label in enumerate(self.channel_name_labels):
                if label.contains(event)[0]:
                    self.edit_channel_name(index)
                    return
        if event.inaxes is None:
            return
        waveform_index = next(
            (index for index, axis in enumerate(self.axes) if event.inaxes is axis),
            None,
        )
        if waveform_index is not None:
            half_range = self.initial_y_half_range
            for index, axis in enumerate(self.axes):
                axis.set_ylim(-half_range, half_range)
                self.y_top_labels[index].set_text(f"+{self._format_scale_value(half_range)}")
                self.y_bottom_labels[index].set_text(f"-{self._format_scale_value(half_range)}")
            self.canvas.draw_idle()
            return

        if event.inaxes is self.spectrum_ax:
            self.user_spectrum_xlim = None
            self.user_spectrum_ylim = None
            self.spectrum_ax.set_xlim(0.2, self.spectrum_max_frequency)
            if self.is_history_mode:
                end = self._history_end_index()
                source = [channel[:end] for channel in self.history_data]
                rate = self.history_rate
            else:
                source = self._current_data_source()
                rate = self.AdsData.sample_rate
            self._update_spectrum(source, rate)
            self.spectrum_canvas.draw_idle()

    def _init_waveform_axes(self):
        """初始化8个独立但共享时间轴的脑电波形子图。"""
        for i, ax in enumerate(self.axes):
            ax.set_facecolor("#0A1729")
            half_range = self.initial_y_half_range
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
            highlight, = ax.plot(
                [], [], color="#FFD34D", linewidth=2.2, alpha=0.95,
                antialiased=True, solid_capstyle="round", zorder=30,
            )
            self.highlight_lines.append(highlight)

            # 左侧幅值参考和通道名称。滚轮调整幅值时同步更新数值。
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

    def refresh_ports(self):
        """扫描串口并标记采集设备，保留当前选择。"""
        current_selection = self.port_combo.currentData()
        self.port_combo.clear()

        ports = list(list_ports.comports())
        ports.sort(key=lambda port: (
            not (port.vid == DEVICE_VID and port.pid == DEVICE_PID), port.device
        ))
        if not ports:
            self.port_combo.addItem("未检测到串口", None)
            self.port_combo.setEnabled(False)
            self.btn_open.setEnabled(False)
            return

        for port in ports:
            target = port.vid == DEVICE_VID and port.pid == DEVICE_PID
            label = f"{port.device}{'  [采集设备]' if target else ''}  {port.description or ''}".strip()
            self.port_combo.addItem(label, port.device)

        self.port_combo.setEnabled(True)
        self.btn_open.setEnabled(True)

        index = self.port_combo.findData(current_selection)
        self.port_combo.setCurrentIndex(index if index >= 0 else 0)

    def _on_wifi_toggled(self, checked):
        index = self.baud_combo.findData(1000000 if checked else USB_BAUDRATE)
        self.baud_combo.setCurrentIndex(index)
        self.baud_combo.setEnabled(bool(checked))
        self.statusBar().showMessage("已切换到 WiFi 透传模式" if checked else "已切换到 USB 模式", 2500)

    def _begin_session_handshake(self):
        self._device_ready = False
        timeout = WIFI_READY_TIMEOUT_S if self.wifi_check.isChecked() else USB_READY_TIMEOUT_S
        self._ready_deadline = time.monotonic() + timeout
        self._queue_commands([
            (ADS1298cmd.enable_device_cmd(), "使能设备", COMMAND_GAP_MS),
            (ADS1298cmd.get_state_cmd(), "查询设备状态", COMMAND_GAP_MS),
        ])
        self.ready_timer.start()

    def _queue_commands(self, entries):
        """替换当前命令链，并保留上一命令的最小发送间隔。"""
        self.command_timer.stop()
        self._pending_commands = list(entries)
        return self._run_next_command()

    def _run_next_command(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self._pending_commands = []
            return False
        if not self._pending_commands:
            return True
        wait_ms = int(max(0, (self._next_command_at - time.monotonic()) * 1000))
        if wait_ms > 0:
            self.command_timer.start(wait_ms + 1)
            return True
        command, description, gap_ms = self._pending_commands.pop(0)
        if not self.send_binary_command(command, description, quiet=True):
            self._pending_commands = []
            if self._starting_stream:
                self._abort_stream_start()
            return False
        if description == "开始采集":
            self._starting_stream = False
        self._next_command_at = time.monotonic() + gap_ms / 1000.0
        if self._pending_commands:
            self.command_timer.start(gap_ms)
        return True

    def _abort_stream_start(self):
        self._finish_auto_recording()
        self._starting_stream = False
        self.is_collecting = False
        self.btn_record.setEnabled(False)
        self.btn_start.setText("▶ 开始采集")
        self.btn_stop.setEnabled(False)
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.set_stream_active(False, "启动命令发送失败")
        if self.motor_imagery_dialog is not None:
            self.motor_imagery_dialog.set_stream_active(False, "启动命令发送失败")

    def _check_device_ready(self):
        if self.AdsData.frames_received > 0:
            self.ready_timer.stop()
            self._device_ready = True
            self.btn_start.setEnabled(True)
            self.btn_impedance.setEnabled(True)
            mode = "WiFi 透传" if self.wifi_check.isChecked() else "USB"
            baud = int(self.baud_combo.currentData() or USB_BAUDRATE)
            self.statusBar().showMessage(f"设备已就绪（{mode} / {baud:,} bps）", 4000)
            return
        if time.monotonic() >= self._ready_deadline:
            self.ready_timer.stop()
            self.btn_start.setEnabled(True)
            hint = ("请检查 WiFi 模块供电与配对状态、COM 口及波特率。"
                    if self.wifi_check.isChecked() else "请检查 USB 连接、驱动与端口选择。")
            self.statusBar().showMessage("未收到设备响应", 5000)
            QMessageBox.warning(self, "设备未响应", hint)

    def _shutdown_device(self):
        """关闭前尽量停止设备，并清除未发送的命令。"""
        self.command_timer.stop()
        self.ready_timer.stop()
        self._pending_commands = []
        self._starting_stream = False
        if self.serial_port is None or not self.serial_port.is_open:
            return
        try:
            if self.is_collecting:
                self.serial_port.write(ADS1298cmd.stop_stream_cmd())
                self.serial_port.flush()
                time.sleep(0.10)
            if self.is_impedance_testing:
                self.serial_port.write(ADS1298cmd.stop_impedance_cmd())
                self.serial_port.flush()
                time.sleep(0.35)
            self.serial_port.write(ADS1298cmd.disable_device_cmd())
            self.serial_port.flush()
            time.sleep(0.35)
        except (serial.SerialException, OSError):
            logger.warning("关闭前清理设备失败", exc_info=True)

    def _reset_connection_ui(self):
        self.ready_timer.stop()
        self.command_timer.stop()
        self._pending_commands = []
        self._starting_stream = False
        self._device_ready = False
        self.btn_open.setText("打开串口")
        self.btn_start.setText("▶ 开始采集")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_impedance.setEnabled(False)
        self.port_combo.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.wifi_check.setEnabled(True)
        self.baud_combo.setEnabled(self.wifi_check.isChecked())
        self.link_status_label.setText("链路：未连接")

    def open_serial(self):
        """打开或关闭当前选择的串口。"""
        if self.serial_port is None or not self.serial_port.is_open:
            port_name = self.port_combo.currentData()
            if not port_name:
                QMessageBox.warning(self, "提示", "请先选择一个有效串口。")
                return
            baudrate = int(self.baud_combo.currentData() or USB_BAUDRATE)
            try:
                self.serial_port = serial.Serial(
                    port=port_name,
                    baudrate=baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.05,
                    write_timeout=1.0,
                )
            except serial.SerialException as exc:
                self.serial_port = None
                tip = "\n请尝试 921600 / 460800 / 115200 bps。" if self.wifi_check.isChecked() and baudrate >= 921600 else ""
                QMessageBox.critical(self, "串口错误", f"打开串口失败：\n{exc}{tip}")
                return

            try:
                self.serial_port.reset_input_buffer()
            except (serial.SerialException, OSError):
                pass
            self.AdsData.reset_link_counters()
            self._link = StreamLossTracker(self.AdsData.sample_rate)
            self._device_base = None
            self._device_report_valid = False
            self._last_eeg_host_time = None
            self.btn_open.setText("关闭串口")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(self.is_collecting)
            self.btn_impedance.setEnabled(False)
            self.port_combo.setEnabled(False)
            self.btn_refresh.setEnabled(False)
            self.wifi_check.setEnabled(False)
            self.baud_combo.setEnabled(False)
            mode = "WiFi 透传" if self.wifi_check.isChecked() else "USB"
            logger.info("打开串口成功：%s @ %d bps（%s）", port_name, baudrate, mode)
            self.statusBar().showMessage(f"已打开 {port_name} @ {baudrate:,} bps，等待设备响应")
            self._begin_session_handshake()
            return

        self._shutdown_device()
        was_collecting = self.is_collecting
        self.is_collecting = False
        self.is_impedance_testing = False
        self.stop_collection()
        if was_collecting:
            self._reset_subject_for_next_session()
        self.stop_impedance_test()
        try:
            self.serial_port.close()
        except serial.SerialException as exc:
            QMessageBox.warning(self, "串口提示", f"关闭串口时发生错误：\n{exc}")
        finally:
            self.serial_port = None
            self._reset_connection_ui()

    def toggle_impedance_test(self, enabled):
        if enabled:
            if self.is_collecting:
                self.btn_impedance.setChecked(False)
                QMessageBox.warning(self, "提示", "请先停止 EEG 采集，再进行阻抗检测。")
                return
            if not self.send_binary_command(ADS1298cmd.impedance_cmd(), "阻抗检测"):
                self.btn_impedance.setChecked(False)
                return
            self.is_impedance_testing = True
            self.btn_impedance.setText("停止阻抗检测")
            self.statusBar().showMessage("阻抗检测已启动", 2500)
        else:
            self.stop_impedance_test()

    def stop_impedance_test(self):
        if self.is_impedance_testing and self.serial_port is not None and self.serial_port.is_open:
            self.send_binary_command(ADS1298cmd.stop_impedance_cmd(), "停止阻抗检测")
        self.is_impedance_testing = False
        if hasattr(self, "btn_impedance"):
            self.btn_impedance.blockSignals(True)
            self.btn_impedance.setChecked(False)
            self.btn_impedance.setText("阻抗检测")
            self.btn_impedance.blockSignals(False)

    def _update_impedance_display(self):
        values = self.AdsData.impedance_data
        if not values:
            return
        for index, (label, value) in enumerate(zip(self.impedance_labels, values)):
            label.setText(f"{self.channel_names[index]}\n{value} kΩ")
        self.statusBar().showMessage("已收到阻抗检测结果", 1200)

    def _update_gravity_display(self):
        acceleration = self.AdsData.acceleration
        if acceleration is None:
            return
        x, y, z = acceleration
        magnitude = float(np.sqrt(x * x + y * y + z * z))
        for label, axis, value in zip(self.gravity_labels[:3], ("X", "Y", "Z"), acceleration):
            label.setText(f"加速度 {axis}: {value:+.3f} g")
        self.gravity_labels[3].setText(f"加速度 |G|: {magnitude:.3f} g")
        if self.AdsData.motion_state is not None:
            self.gravity_state_label.setText(f"状态: {self.AdsData.motion_state}")

    def toggle_serial_log(self, enabled):
        self.serial_log_enabled = bool(enabled)
        if self.serial_log_dialog is None:
            self.serial_log_dialog = SerialLogDialog(self)
            for direction, payload in self.serial_log_entries:
                self.serial_log_dialog.append_entry(direction, payload)

        if self.serial_log_enabled:
            self.serial_log_dialog.show()
            self.serial_log_dialog.raise_()
            self.serial_log_dialog.activateWindow()
            self.btn_serial_log.setText("关闭串口日志")
            self.statusBar().showMessage("串口日志已开启", 2500)
        else:
            self.serial_log_dialog.hide()
            self.btn_serial_log.setText("串口日志")
            self.statusBar().showMessage("串口日志已关闭", 2500)

    def _append_serial_log(self, direction, payload):
        if not self.serial_log_enabled or not payload:
            return
        payload = bytes(payload)
        self.serial_log_entries.append((direction, payload))
        if len(self.serial_log_entries) > 200:
            del self.serial_log_entries[:-200]
        if self.serial_log_dialog is not None:
            self.serial_log_dialog.append_entry(direction, payload)

    def send_protocol_command(self, command_text):
        """发送对应的 USB 协议指令，并在状态栏显示回显。"""
        if self.serial_port is None or not self.serial_port.is_open:
            QMessageBox.warning(self, "提示", "请先打开串口。")
            return

        try:
            payload = (command_text + "\r\n").encode("ascii")
            self.serial_port.write(payload)
            self._append_serial_log("TX", payload)
            logger.debug(f"已发送协议命令: {command_text} -> {payload!r}")
        except serial.SerialException as exc:
            logger.exception("发送协议命令失败")
            QMessageBox.critical(self, "串口错误", f"发送协议命令失败：\n{exc}")
            return

        self.statusBar().showMessage(f"已发送：{command_text}", 2500)

    def send_binary_command(self, command_bytes, description, quiet=False):
        """Send one main2.txt BP frame over USB CDC."""
        if self.serial_port is None or not self.serial_port.is_open:
            QMessageBox.warning(self, "提示", "请先打开串口。")
            return False

        try:
            self.serial_port.write(bytes(command_bytes))
            self._append_serial_log("TX", command_bytes)
            self.serial_port.flush()
            logger.debug("发送二进制协议命令 %s -> %r", description, command_bytes)
        except (serial.SerialException, OSError) as exc:
            logger.exception("发送二进制协议命令失败")
            QMessageBox.critical(self, "串口错误", f"发送{description}失败：\n{exc}")
            return False

        if not quiet:
            self.statusBar().showMessage(f"已发送：{description}", 2500)
        return True

    def open_alpha_model(self):
        if self.alpha_model_dialog is None:
            self.alpha_model_dialog = AlphaModelDialog(self.channel_names, self)
        dialog = self.alpha_model_dialog
        if not dialog.isVisible():
            dialog.start_session()
            dialog.set_available_channels(self.channel_states)
            active = self.is_collecting and not self.is_history_mode
            reason = "历史回访期间暂停脑电控制" if self.is_history_mode else "请先开始 EEG 采集"
            dialog.set_stream_active(active, reason)
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def open_motor_imagery(self):
        if self.motor_imagery_dialog is None:
            self.motor_imagery_dialog = MotorImageryDialog(self.channel_names, self)
        dialog = self.motor_imagery_dialog
        if not dialog.isVisible():
            dialog.start_session()
            dialog.set_available_channels(self.channel_states)
            active = self.is_collecting and not self.is_history_mode
            reason = "历史回访期间暂停运动想象控制" if self.is_history_mode else "请先开始 EEG 采集"
            dialog.set_stream_active(active, reason)
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def open_rhythm_analysis(self):
        if self.rhythm_analysis_dialog is None:
            self.rhythm_analysis_dialog = RhythmAnalysisDialog(self)
        self.rhythm_analysis_dialog.show()
        self.rhythm_analysis_dialog.raise_()
        self.rhythm_analysis_dialog.activateWindow()

    def start_collect(self):
        """切换开始或停止采集状态。"""
        if not self.is_collecting and self.subject_info is None:
            QMessageBox.warning(self, "提示", "未录入被采集者信息")
            return
        if self.serial_port is None or not self.serial_port.is_open:
            QMessageBox.warning(self, "提示", "请先打开串口。")
            return

        if self.is_impedance_testing:
            QMessageBox.warning(self, "提示", "请先停止阻抗检测，再开始 EEG 采集。")
            return

        if self.is_history_mode:
            return

        if not self.is_collecting:
            self.AdsData.clear()
            self.AdsData.reset_link_counters()
            self.AdsData.sample_rate = 250
            self._reset_filter_runtime()
            self._clear_waveform_lines()
            self._link = StreamLossTracker(250)
            now = time.monotonic()
            self._stats_started = now
            self._next_status_poll = now + STATUS_POLL_INTERVAL_S
            self._last_eeg_host_time = None
            self._device_base = None
            self._device_drops = 0
            self._device_link_errors = 0
            self._device_report_valid = False
            self.ready_timer.stop()
            self._starting_stream = True
            if not self._queue_commands([
                (ADS1298cmd.enable_device_cmd(), "使能设备", COMMAND_GAP_MS),
                (ADS1298cmd.set_config_cmd(
                    sample_rate=250,
                    gain=24,
                    channel_mask=0xFF,
                    input_type=0,
                    bias_enabled=1,
                    test_signal=0,
                    duration_sec=0,
                ), "采样配置", COMMAND_GAP_MS),
                (ADS1298cmd.get_state_cmd(), "查询设备状态", COMMAND_GAP_MS),
                (ADS1298cmd.start_stream_cmd(), "开始采集", COMMAND_GAP_MS),
            ]):
                return
            self.eye_blink_detector.reset()
            self.is_collecting = True
            self.collection_subject = dict(self.subject_info)
            self.collection_started_at = datetime.datetime.now()
            self.recorded_files = []
            self.btn_subject.setEnabled(False)
            self.btn_record.setEnabled(True)
            self.record_status_label.setText("未保存数据")
            if self.alpha_model_dialog is not None:
                self.alpha_model_dialog.start_session()
                self.alpha_model_dialog.set_stream_active(True)
            if self.motor_imagery_dialog is not None:
                self.motor_imagery_dialog.start_session()
                self.motor_imagery_dialog.set_stream_active(True)
            self.btn_start.setText("■ 停止采集")
            self.btn_stop.setEnabled(True)
            return
        else:
            self.stop_collection()

    def stop_collection(self):
        """停止采集，但保留当前缓存数据用于显示和导出。"""
        was_collecting = self.is_collecting
        self._finish_auto_recording()
        self.btn_record.setEnabled(False)
        if was_collecting:
            self._reset_subject_for_next_session()
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.set_stream_active(False, "采集已停止，模型已冻结")
        if self.motor_imagery_dialog is not None:
            self.motor_imagery_dialog.set_stream_active(False, "采集已停止，运动想象控制已暂停")
        if self.is_collecting and self.serial_port is not None and self.serial_port.is_open:
            self._starting_stream = False
            self._queue_commands([
                (ADS1298cmd.stop_stream_cmd(), "停止采集", SLOW_COMMAND_GAP_MS),
            ])
            self.is_collecting = False
            if hasattr(self, "btn_start"):
                self.btn_start.setText("▶ 开始采集")
                self.btn_stop.setEnabled(False)
            return

        self.is_collecting = False
        if hasattr(self, "btn_start"):
            self.btn_start.setText("▶ 开始采集")
            self.btn_stop.setEnabled(False)

    def _reset_subject_for_next_session(self):
        # 已采数据仍保留本次身份供手动导出，下次采集需要重新核对被采集者。
        self.subject_info = None
        self.subject_status_label.setText("未录入被采集者信息")
        self.btn_subject.setEnabled(True)

    def _reset_filter_runtime(self):
        """清空滤波输出与IIR历史状态，但不改变按钮开关状态。"""
        self.eeg_filter.reset()
        self.filter_sample_rate = None
        self.filtered_source_length = 0
        self.filtered_emg_data = [[] for _ in range(8)]
        self.display_plot_source = None

    def _live_plot_sources(self, display_source, rate):
        """Keep a bounded plot window while retaining complete data for CSV export."""
        if rate <= 0:
            return display_source, self.AdsData.emg_data, 0
        capacity = max(1, int(rate * (self.display_seconds + 3)))
        if (self.raw_plot_buffer is None or self.plot_buffer_rate != rate
                or self.raw_plot_buffer.capacity != capacity):
            self.raw_plot_buffer = MultiChannelRingBuffer(capacity)
            self.display_plot_buffer = MultiChannelRingBuffer(capacity)
            self.plot_buffer_rate = rate
            self.display_plot_source = None

        self._sync_plot_buffer(self.raw_plot_buffer, self.AdsData.emg_data)
        if display_source is self.AdsData.emg_data:
            display_buffer = self.raw_plot_buffer
        else:
            if display_source is not self.display_plot_source:
                self.display_plot_buffer.clear()
                self.display_plot_source = display_source
            self._sync_plot_buffer(self.display_plot_buffer, display_source)
            display_buffer = self.display_plot_buffer

        display_values = display_buffer.snapshot()
        raw_values = self.raw_plot_buffer.snapshot()
        offset = display_buffer.source_length - len(display_values)
        raw_offset = self.raw_plot_buffer.source_length - len(raw_values)
        if offset != raw_offset or len(display_values) != len(raw_values):
            # A filter rebuild may momentarily lag behind the decoder.
            return display_source, self.AdsData.emg_data, 0
        return display_values.T, raw_values.T, offset

    @staticmethod
    def _sync_plot_buffer(buffer, source):
        length = min(map(len, source[:8])) if len(source) >= 8 else 0
        if length < buffer.source_length:
            buffer.clear()
        if length > buffer.source_length:
            start = buffer.source_length
            if length - start > buffer.capacity:
                buffer.clear()
                start = length - buffer.capacity
            block = np.column_stack([
                np.asarray(channel[start:length], dtype=np.float64)
                for channel in source[:8]
            ])
            buffer.append(block)
            buffer.source_length = length

    def _common_raw_length(self):
        """返回八个原始通道均已收到的公共采样点数。"""
        if len(self.AdsData.emg_data) < 8:
            return 0
        return min(len(self.AdsData.emg_data[index]) for index in range(8))

    def _set_filter_button_state(self, enabled):
        """滤波配置不可用时关闭各开关。"""
        self.filter_enabled = bool(enabled)
        self.btn_filter.blockSignals(True)
        self.btn_filter.setChecked(self.filter_enabled)
        self.btn_filter.blockSignals(False)
        if not enabled:
            self.notch_checkbox.blockSignals(True)
            self.notch_checkbox.setChecked(False)
            self.notch_checkbox.blockSignals(False)
            self.dc_checkbox.blockSignals(True)
            self.dc_checkbox.setChecked(False)
            self.dc_checkbox.blockSignals(False)

    def toggle_filter(self, checked):
        """打开或关闭可调带通滤波。"""
        self._filter_settings_changed()

    def _filter_settings_changed(self, *_args):
        """参数改变后重算现有数据，并继续使用新的滤波器状态。"""
        # 先保存旧数据源的尾批，切换后只记录此后的采样，避免重算导致重复行。
        if self._recorder is not None:
            self._append_new_filtered_data()
            self._record_new_samples()
        low, high = self.low_cutoff.value(), self.high_cutoff.value()
        if low >= high:
            if self.sender() is self.low_cutoff:
                self.high_cutoff.blockSignals(True)
                self.high_cutoff.setValue(min(self.high_cutoff.maximum(), low + 0.1))
                self.high_cutoff.blockSignals(False)
            else:
                self.low_cutoff.blockSignals(True)
                self.low_cutoff.setValue(max(self.low_cutoff.minimum(), high - 0.1))
                self.low_cutoff.blockSignals(False)
        self.filter_enabled = (
            self.btn_filter.isChecked()
            or self.notch_checkbox.isChecked()
            or self.dc_checkbox.isChecked()
        )
        if not self.filter_enabled:
            self._reset_filter_runtime()
            self._refresh_all_graphs(self.AdsData.emg_data)
            return
        try:
            self._rebuild_filtered_data()
        except (ValueError, FloatingPointError) as exc:
            self._set_filter_button_state(False)
            self._reset_filter_runtime()
            self._refresh_all_graphs(self.AdsData.emg_data)
            QMessageBox.warning(self, "滤波器错误", f"滤波参数无效：\n{exc}")
            return
        self._refresh_all_graphs(self.filtered_emg_data)

    def _rebuild_filtered_data(self):
        """从当前原始缓存重新生成滤波数据，并建立后续实时滤波状态。"""
        self._reset_filter_runtime()
        rate = int(self.AdsData.sample_rate)
        self.eeg_filter.configure(
            rate, self.low_cutoff.value(), self.high_cutoff.value(),
            self.btn_filter.isChecked(), self.notch_checkbox.isChecked(),
            self.dc_checkbox.isChecked(),
        )
        self.filter_sample_rate = rate
        raw_length = self._common_raw_length()
        if raw_length <= 0:
            return

        raw_block = np.vstack(
            [
                np.asarray(self.AdsData.emg_data[index][:raw_length], dtype=np.float64)
                for index in range(8)
            ]
        )
        filtered_block = self.eeg_filter.data_filter(rate, raw_block, axis=-1)

        self.filtered_emg_data = [filtered_block[index].tolist() for index in range(8)]
        self.filtered_source_length = raw_length
        self.filter_sample_rate = rate

    def _append_new_filtered_data(self):
        """只对上次处理后新增的八通道采样点执行实时滤波。"""
        if not self.filter_enabled:
            return

        raw_length = self._common_raw_length()
        if raw_length <= 0:
            return

        rate = int(self.AdsData.sample_rate)
        # 原始缓存被清空、长度回退或采样率改变时，必须重新建立状态。
        if (
            self.eeg_filter.settings is None
            or self.filter_sample_rate is None
            or raw_length < self.filtered_source_length
            or self.filter_sample_rate not in (None, rate)
        ):
            try:
                self._rebuild_filtered_data()
            except ValueError as exc:
                self._set_filter_button_state(False)
                self._reset_filter_runtime()
                QMessageBox.warning(self, "滤波器提示", str(exc))
            return

        start_index = self.filtered_source_length
        if raw_length <= start_index:
            return

        raw_block = np.vstack(
            [
                np.asarray(
                    self.AdsData.emg_data[index][start_index:raw_length],
                    dtype=np.float64,
                )
                for index in range(8)
            ]
        )
        filtered_block = self.eeg_filter.data_filter(rate, raw_block, axis=-1)

        for channel_idx in range(8):
            self.filtered_emg_data[channel_idx].extend(
                filtered_block[channel_idx].tolist()
            )

        self.filtered_source_length = raw_length
        self.filter_sample_rate = rate

    def _current_data_source(self):
        """返回当前界面应显示和导出的原始或滤波数据。"""
        if self.filter_enabled:
            return self.filtered_emg_data
        return self.AdsData.emg_data

    def open_history(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 EEG 历史数据", "", "CSV 文件 (*.csv)"
        )
        if not path:
            return
        try:
            data = read_history_csv(path, self.channel_names, self.channel_name_history)
        except (OSError, UnicodeError, ValueError) as exc:
            QMessageBox.warning(self, "历史数据读取失败", str(exc))
            return

        rate = sample_rate_from_filename(path)
        if rate is None:
            rate, confirmed = QInputDialog.getInt(
                self, "历史数据采样率", "采样率 (Hz)",
                int(self.AdsData.sample_rate or 250), 80, 4000,
            )
            if not confirmed:
                return
        if rate < 80 or rate > 4000:
            QMessageBox.warning(self, "历史数据读取失败", "采样率须在 80–4000 Hz 之间")
            return

        if self.is_collecting:
            self.stop_collection()
        self.stop_impedance_test()
        self.history_timer.stop()
        self.btn_history_play.setText("▶")
        self.history_data = data
        self.history_rate = rate
        self.history_file_path = path
        self.is_history_mode = True
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.set_stream_active(False, "历史回访期间暂停脑电控制")
        if self.motor_imagery_dialog is not None:
            self.motor_imagery_dialog.set_stream_active(False, "历史回访期间暂停运动想象控制")
        self.history_controls.setVisible(True)
        self.btn_live.setVisible(True)
        self.eye_blink_status.setVisible(False)
        self.waveform_mode_title.setText("EEG 历史波形")
        self.spectrum_title.setText("EEG 频谱（回访） 频率（Hz）")
        self.btn_start.setEnabled(False)
        self.btn_impedance.setEnabled(False)
        for control in (self.btn_filter, self.low_cutoff, self.high_cutoff,
                        self.notch_checkbox, self.dc_checkbox):
            control.setEnabled(False)

        duration = len(data[0]) / float(rate)
        maximum = max(1, int(np.ceil(duration * 10)))
        self.history_slider.blockSignals(True)
        self.history_slider.setRange(0, maximum)
        self.history_slider.setValue(min(maximum, int(round(self.display_seconds * 10))))
        self.history_slider.blockSignals(False)
        self._render_history()
        self.statusBar().showMessage(f"历史回访：{path}", 5000)

    def leave_history(self):
        if not self.is_history_mode:
            return
        self.history_timer.stop()
        self.btn_history_play.setText("▶")
        self.is_history_mode = False
        self.history_data = None
        self.history_file_path = ""
        self.history_controls.setVisible(False)
        self.btn_live.setVisible(False)
        self.eye_blink_status.setVisible(True)
        self.waveform_mode_title.setText("EEG 实时波形")
        self.spectrum_title.setText("EEG 频谱（实时） 频率（Hz）")
        serial_open = self.serial_port is not None and self.serial_port.is_open
        self.btn_start.setEnabled(serial_open)
        self.btn_impedance.setEnabled(serial_open)
        for control in (self.btn_filter, self.low_cutoff, self.high_cutoff,
                        self.notch_checkbox, self.dc_checkbox):
            control.setEnabled(True)
        self._refresh_all_graphs()

    def _history_end_index(self):
        if self.history_data is None:
            return 0
        return min(len(self.history_data[0]), int(round(
            self.history_slider.value() * self.history_rate / 10.0
        )))

    @staticmethod
    def _format_history_time(seconds):
        minutes, remaining = divmod(max(0.0, seconds), 60.0)
        return f"{int(minutes):02d}:{remaining:04.1f}"

    def _render_history(self, _value=None):
        if not self.is_history_mode or self.history_data is None:
            return
        end = self._history_end_index()
        active_count = 0
        for index, channel in enumerate(self.history_data):
            active_count += bool(self.update_graph(index, channel, self.history_rate, end))
        self._update_alpha_status(active_count)
        self._update_spectrum([channel[:end] for channel in self.history_data], self.history_rate)
        elapsed = end / float(self.history_rate)
        total = len(self.history_data[0]) / float(self.history_rate)
        self.history_time_label.setText(
            f"{self._format_history_time(elapsed)} / {self._format_history_time(total)}"
        )
        self.canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def toggle_history_playback(self):
        if not self.is_history_mode:
            return
        if self.history_timer.isActive():
            self.history_timer.stop()
            self.btn_history_play.setText("▶")
            return
        if self.history_slider.value() >= self.history_slider.maximum():
            self.history_slider.setValue(min(
                self.history_slider.maximum(), int(round(self.display_seconds * 10))
            ))
        self.history_timer.start(100)
        self.btn_history_play.setText("Ⅱ")

    def _advance_history(self):
        next_value = self.history_slider.value() + 1
        if next_value > self.history_slider.maximum():
            self.history_timer.stop()
            self.btn_history_play.setText("▶")
            return
        self.history_slider.setValue(next_value)

    def _track_alpha_occurrences(self, activity, sample_count):
        if sample_count <= self.alpha_last_sample_count:
            return
        self.alpha_last_sample_count = sample_count
        active = [bool(value) for value in activity]
        new_count = sum(
            current and not previous
            for current, previous in zip(active, self.alpha_previous_activity)
        )
        self.alpha_previous_activity = active
        if new_count:
            self.alpha_occurrence_count += new_count
            if not self.alpha_reset_timer.isActive():
                self.alpha_reset_timer.start(180_000)

    def _reset_alpha_occurrences(self):
        self.alpha_reset_timer.stop()
        self.alpha_occurrence_count = 0
        if not self.is_history_mode:
            self._update_alpha_status(self.alpha_live_active_count)

    def _track_eye_blinks(self, new_raw_samples, sample_rate):
        if not self.is_collecting or self.is_history_mode or not new_raw_samples:
            return
        new_count = self.eye_blink_detector.process(new_raw_samples, sample_rate)
        if new_count:
            self.eye_blink_count += new_count
            if not self.eye_blink_reset_timer.isActive():
                self.eye_blink_reset_timer.start(180_000)
            self._update_eye_blink_status()

    def _reset_eye_blink_count(self):
        self.eye_blink_reset_timer.stop()
        self.eye_blink_count = 0
        self._update_eye_blink_status()

    def _update_eye_blink_status(self):
        self.eye_blink_status.setText(
            f"睁闭眼估计 3分累计{self.eye_blink_count}次"
        )

    def _update_alpha_status(self, active_count):
        low, high = self.detection_band
        band_text = f"{low:g}–{high:g} Hz"
        if self.is_history_mode:
            self.alpha_status.setText(f"{band_text}: {active_count}/8（回访）")
            return
        self.alpha_live_active_count = active_count
        current = "--" if active_count is None else f"{active_count}/8"
        self.alpha_status.setText(
            f"{band_text} {current} · 3分累计{self.alpha_occurrence_count}次"
        )

    def _on_detection_band_changed(self):
        low, high = self.band_low.value(), self.band_high.value()
        if high - low < 0.5:
            width = self.detection_band[1] - self.detection_band[0]
            if self.sender() is self.band_low:
                high = min(40.0, low + width)
                with QSignalBlocker(self.band_high):
                    self.band_high.setValue(high)
            elif self.sender() is self.band_high:
                low = max(1.0, high - width)
                with QSignalBlocker(self.band_low):
                    self.band_low.setValue(low)
            else:
                QMessageBox.warning(self, "频段无效", "上限必须至少比下限高 0.5 Hz。")
                self.band_low.setValue(self.detection_band[0])
                self.band_high.setValue(self.detection_band[1])
                return
        if (low, high) == self.detection_band:
            return
        self.detection_band = (low, high)
        self.alpha_reset_timer.stop()
        self.alpha_occurrence_count = 0
        self.alpha_previous_activity = [False] * 8
        self.alpha_last_sample_count = self._common_raw_length()
        self._refresh_all_graphs()

    def _refresh_all_graphs(self, data_source=None):
        """立即使用指定数据源刷新八个通道。"""
        if not hasattr(self, "lines") or len(self.lines) < 8:
            return
        if self.is_history_mode:
            self._render_history()
            return
        source = data_source if data_source is not None else self._current_data_source()
        rate = int(getattr(self.AdsData, "sample_rate", 0) or 0)
        plot_source, raw_source, sample_offset = self._live_plot_sources(source, rate)
        active_count = 0
        for channel_idx in range(8):
            channel_data = plot_source[channel_idx] if channel_idx < len(plot_source) else []
            raw_data = raw_source[channel_idx]
            active_count += bool(self.update_graph(
                channel_idx, channel_data, rate, detection_data=raw_data,
                sample_offset=sample_offset,
            ))
        self._update_alpha_status(active_count)
        self._update_spectrum(plot_source, rate)
        self.canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def _clear_waveform_lines(self):
        """清空波形，并把每个通道的量程恢复到初始值。"""
        self.alpha_previous_activity = [False] * 8
        self.alpha_last_sample_count = 0
        self.plot_buffer_rate = None
        self.raw_plot_buffer = None
        self.display_plot_buffer = None
        self.display_plot_source = None
        self.user_spectrum_xlim = None
        self.user_spectrum_ylim = None
        for channel_idx, line in enumerate(self.lines):
            line.set_data([], [])
            self.highlight_lines[channel_idx].set_data([], [])
            self.axes[channel_idx].set_ylim(
                -self.initial_y_half_range,
                self.initial_y_half_range,
            )
            value_text = self._format_scale_value(self.initial_y_half_range)
            self.y_top_labels[channel_idx].set_text(f"+{value_text}")
            self.y_bottom_labels[channel_idx].set_text(f"-{value_text}")

        self.axes[-1].set_xlim(0.0, self.display_seconds)
        self._update_alpha_status(None)
        self.canvas.draw_idle()
        for line in self.spectrum_lines:
            line.set_data([], [])
        self.spectrum_ax.set_xlim(0.2, self.spectrum_max_frequency)
        self.spectrum_ax.set_ylim(-100.0, 25.0)
        self._update_spectrum_peak_annotation()
        self.spectrum_canvas.draw_idle()

    def toggle_channel(self, state, channel_idx):
        """选中时高亮通道；取消选中时保留波形但降低亮度。"""
        enabled = bool(state)
        self.channel_states[channel_idx] = enabled
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.set_available_channels(self.channel_states)
        if self.motor_imagery_dialog is not None:
            self.motor_imagery_dialog.set_available_channels(self.channel_states)

        line = self.lines[channel_idx]
        line.set_visible(True)
        if enabled:
            line.set_alpha(1.0)
            line.set_linewidth(0.90)
            line.set_zorder(20)
            self.highlight_lines[channel_idx].set_alpha(0.95)
        else:
            line.set_alpha(0.12)
            line.set_linewidth(0.60)
            line.set_zorder(1)
            self.highlight_lines[channel_idx].set_alpha(0.12)

        self._set_color_block_style(channel_idx, enabled)
        if channel_idx < len(self.spectrum_lines):
            self.spectrum_lines[channel_idx].set_visible(enabled)
            self._update_spectrum_peak_annotation()
        self.canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def update_show(self):
        """定时读取串口缓存、解析数据、执行可选滤波并刷新波形。"""
        if self.serial_port is None or not self.serial_port.is_open:
            if self._recorder is not None:
                self._handle_serial_disconnect()
            return

        payload = bytearray()
        deadline = time.monotonic() + 0.03
        try:
            while time.monotonic() < deadline:
                waiting = self.serial_port.in_waiting
                if waiting <= 0:
                    break
                payload.extend(self.serial_port.read(min(waiting, 65536)))
        except (serial.SerialException, OSError) as exc:
            QMessageBox.critical(self, "串口错误", f"读取串口数据失败：\n{exc}")
            self._handle_serial_disconnect()
            return

        if not payload:
            return
        now = time.monotonic()
        if not self.is_collecting or now - self._last_rx_log_time >= 1.0:
            self._append_serial_log("RX", payload[:128])
            self._last_rx_log_time = now

        self.AdsData.parse_data(payload)
        self._consume_link_events()
        self._consume_device_status()
        self._update_impedance_display()
        self._update_gravity_display()
        self._track_eye_blinks(self.AdsData.new_raw_samples, self.AdsData.sample_rate)
        if (self.is_collecting and not self.is_history_mode
                and self.alpha_model_dialog is not None
                and self.alpha_model_dialog.isVisible()):
            self.alpha_model_dialog.feed_samples(
                self.AdsData.new_raw_samples, self.AdsData.sample_rate,
            )
        if (self.is_collecting and not self.is_history_mode
                and self.motor_imagery_dialog is not None
                and self.motor_imagery_dialog.isVisible()):
            self.motor_imagery_dialog.feed_samples(
                self.AdsData.new_raw_samples, self.AdsData.sample_rate,
            )
        if self.is_history_mode:
            return
        if not any(len(channel) > 0 for channel in self.AdsData.emg_data):
            return

        # 只对新追加的数据滤波，不能每100 ms重复处理全部历史缓存。
        self._append_new_filtered_data()
        self._record_new_samples()
        data_source = self._current_data_source()
        plot_source, raw_source, sample_offset = self._live_plot_sources(
            data_source, self.AdsData.sample_rate,
        )

        activity = []
        for channel_idx in range(8):
            activity.append(bool(self.update_graph(
                channel_idx,
                plot_source[channel_idx],
                self.AdsData.sample_rate,
                detection_data=raw_source[channel_idx],
                sample_offset=sample_offset,
            )))
        if self.is_collecting:
            self._track_alpha_occurrences(activity, self._common_raw_length())
        self._update_alpha_status(sum(activity))

        self._update_spectrum(plot_source, self.AdsData.sample_rate)

        self.canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def _consume_link_events(self):
        events = self.AdsData.eeg_frame_events
        for seq, timestamp in events:
            self._link.update(seq, timestamp)
        if events:
            self._last_eeg_host_time = time.monotonic()
            self.AdsData.eeg_frame_events = []

    def _consume_device_status(self):
        status = self.AdsData.device_status
        if not status or not status.get("link_stats"):
            return
        values = (status["ads_drops"], status["wireless_tx_overflow"],
                  status["wireless_rx_overflow"])
        if self._device_base is None:
            self._device_base = values
        self._device_drops = max(0, values[0] - self._device_base[0])
        self._device_link_errors = (
            max(0, values[1] - self._device_base[1])
            + max(0, values[2] - self._device_base[2])
        )
        self._device_report_valid = True

    def _update_link_display(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.link_status_label.setText("链路：未连接")
            return
        now = time.monotonic()
        received = self.AdsData.eeg_frames_received
        elapsed = now - self._stats_started if self._stats_started else 0.0
        fps = received / elapsed if elapsed > 0 else 0.0
        rate = self._link.loss_rate(received)
        parts = [
            f"EEG {fps:.1f} fps", f"丢包 {self._link.lost_by_seq} ({rate * 100:.2f}%)",
            f"时戳丢样 {self._link.lost_by_time}", f"断流 {self._link.outages}",
            f"CRC {self.AdsData.crc_errors}",
        ]
        if self._device_report_valid:
            parts.extend((f"设备丢 {self._device_drops}",
                          f"链路错误 {self._device_link_errors}"))
        if self.is_collecting:
            if self._last_eeg_host_time is None or now - self._last_eeg_host_time > STREAM_TIMEOUT_S:
                parts.insert(0, "未收到 EEG 数据")
            if now >= self._next_status_poll and not self._pending_commands:
                self._next_status_poll = now + STATUS_POLL_INTERVAL_S
                self._queue_commands([
                    (ADS1298cmd.get_state_cmd(), "查询设备状态", COMMAND_GAP_MS),
                ])
        self.link_status_label.setText("链路：" + " · ".join(parts))

    def update_graph(self, channel_idx, data, rate, end_index=None,
                     detection_data=None, sample_offset=0):
        """更新指定通道的 10 秒波形和所选频段高亮。"""
        if rate <= 0 or len(data) == 0:
            self.lines[channel_idx].set_data([], [])
            self.highlight_lines[channel_idx].set_data([], [])
            return False

        display_length = int(rate * self.display_seconds)
        end_index = len(data) if end_index is None else min(len(data), int(end_index))
        if end_index <= 0:
            self.lines[channel_idx].set_data([], [])
            self.highlight_lines[channel_idx].set_data([], [])
            self.axes[-1].set_xlim(0.0, self.display_seconds)
            return False
        start_index = max(0, end_index - display_length)

        y_data = np.asarray(data[start_index:end_index], dtype=np.float64)
        x_data = np.arange(
            sample_offset + start_index, sample_offset + end_index,
            dtype=np.float64,
        ) / float(rate)
        analysis_source = data if detection_data is None else detection_data
        if len(analysis_source) < end_index:
            analysis_source = data
        hop_size = max(1, int(round(0.5 * rate)))
        absolute_start = sample_offset + start_index
        analysis_start = max(sample_offset, (
            (absolute_start - int(round(2 * rate))) // hop_size
        ) * hop_size)
        analysis_start -= sample_offset
        analysis_data = analysis_source[analysis_start:end_index]
        activity = alpha_activity_mask(
            analysis_data, rate, sample_offset=sample_offset + analysis_start,
            band=self.detection_band,
        )[start_index - analysis_start:]

        # 电极的直流偏置可能达到数万 uV。只在波形显示时去掉当前窗口的
        # 中位数，让纵轴反映脑电波动；采集缓存、滤波输入和 CSV 均保留原值。
        finite_y_data = y_data[np.isfinite(y_data)]
        baseline = float(np.median(finite_y_data)) if finite_y_data.size else 0.0
        y_data = y_data - baseline
        highlighted_y = np.where(activity, y_data, np.nan)

        # 对显示数据进行抽点，降低高采样率下的绘图负载；原始缓存不变。
        if y_data.size > self.max_display_points:
            step = int(np.ceil(y_data.size / self.max_display_points))
            y_data = y_data[::step]
            highlighted_y = highlighted_y[::step]
            x_data = x_data[::step]

        self.lines[channel_idx].set_data(x_data, y_data)
        self.highlight_lines[channel_idx].set_data(x_data, highlighted_y)

        # 统一更新共享时间轴，形成向左滚动的固定时间窗。
        current_time = (sample_offset + end_index) / float(rate)
        if current_time < self.display_seconds:
            x_min = 0.0
            x_max = self.display_seconds
        else:
            x_min = current_time - self.display_seconds
            x_max = current_time
        self.axes[-1].set_xlim(x_min, x_max)
        return bool(np.any(activity[-max(1, int(rate)):]))

    def _handle_serial_disconnect(self):
        """串口异常断开后的界面复位。"""
        logger.warning("串口异常断开，界面将复位")
        self._finish_auto_recording()
        if self.is_collecting:
            self._reset_subject_for_next_session()
        self.is_collecting = False
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.set_stream_active(False, "串口已断开，模型已冻结")
        if self.motor_imagery_dialog is not None:
            self.motor_imagery_dialog.set_stream_active(False, "串口已断开，运动想象控制已暂停")
        self.is_impedance_testing = False
        self.stop_impedance_test()
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self._reset_connection_ui()
        self.statusBar().showMessage("串口已断开，界面已复位", 3000)

    def select_record_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择 EEG 自动记录目录（按日期建立子目录）", self.record_directory,
        )
        if directory:
            self.record_directory = directory
            self.record_settings.setValue("directory", directory)
            self.btn_record_directory.setToolTip(directory)

    def edit_subject(self):
        if self.is_collecting:
            return
        dialog = SubjectDialog(self, self.subject_info)
        if dialog.exec_() == QDialog.Accepted:
            self.subject_info = dialog.subject()
            label = "宠物" if self.subject_info["kind"] == "pet" else "人"
            self.subject_status_label.setText(f"{label}：{self.subject_info['name']}")

    def toggle_recording(self):
        if self._recorder is not None:
            self._finish_auto_recording()
            return
        if not self.is_collecting or self.is_history_mode:
            return
        if not self.record_settings.contains("directory"):
            self.select_record_directory()
            self.record_settings.setValue("directory", self.record_directory)
        self._start_auto_recording()

    def _save_record_limits(self):
        self.record_settings.setValue("segment_seconds", self.record_seconds.value())
        self.record_settings.setValue("max_rows", self.record_max_rows.value())

    def _start_auto_recording(self):
        if not self.is_collecting or self.is_history_mode or self._recorder is not None:
            return
        self._append_new_filtered_data()
        # 保存开关只控制此刻之后的数据，不能补写此前未选择保存的采样。
        self._record_cursor = self._common_raw_length()
        self._record_error_reported = False
        token = self._record_token = object()
        self._recorder = SegmentedCsvRecorder(
            self.record_directory, self.record_seconds.value(), self.record_max_rows.value(),
            on_path=lambda path: self.recording_path.emit(token, path),
            on_error=lambda error: self.recording_error.emit(token, error),
            started_at=self.collection_started_at, subject=self.collection_subject,
        )
        self.btn_record.setText("结束保存数据")
        for widget in (self.btn_record_directory, self.record_seconds, self.record_max_rows):
            widget.setEnabled(False)
        self.record_status_label.setText("自动记录：等待采样")
        self.record_status_label.setToolTip(self.record_directory)

    def _record_new_samples(self):
        if self._recorder is None or self.is_history_mode:
            return
        source = self._current_data_source()
        length = min((len(channel) for channel in source), default=0)
        while self._record_cursor < length:
            end = min(length, self._record_cursor + 256)
            rows = list(zip(*(channel[self._record_cursor:end] for channel in source)))
            accepted = self._recorder.submit(
                rows, self.AdsData.sample_rate,
                "filtered" if self.filter_enabled else "raw", self.channel_names,
            )
            self._record_cursor = end
            if not accepted:
                break

    def _on_recording_path(self, token, path):
        if token is self._record_token and not self._record_error_reported:
            self.record_status_label.setText(f"正在记录：{path}")
            self.record_status_label.setToolTip(path)

    def _on_recording_error(self, token, message):
        if token is not self._record_token or self._record_error_reported:
            return
        self._record_error_reported = True
        logger.error("自动记录失败，采集继续：%s", message)
        self.record_status_label.setText("自动记录失败，采集继续")
        self.record_status_label.setToolTip(message)
        self.statusBar().showMessage(f"自动记录失败，采集继续：{message}", 10000)

    def _finish_auto_recording(self):
        recorder = self._recorder
        if recorder is None:
            return
        self._append_new_filtered_data()
        self._record_new_samples()
        self.recorded_files.extend(recorder.close())
        if recorder.error:
            self._on_recording_error(self._record_token, recorder.error)
        self._recorder = None
        self._record_token = None
        self.btn_record.setText("开始保存数据")
        for widget in (self.btn_record_directory, self.record_seconds, self.record_max_rows):
            widget.setEnabled(True)
        summary = f"本次记录已结束，共 {len(self.recorded_files)} 个文件"
        if recorder.error:
            summary += "（记录失败，文件可能不完整）"
        paths = "\n".join(self.recorded_files)
        self.record_status_label.setText(summary + "（悬停查看分段列表）")
        self.record_status_label.setToolTip(summary + "\n" + paths)
        self.statusBar().showMessage(summary + "：" + "；".join(self.recorded_files), 15000)
        logger.info("%s\n%s", summary, paths)
        if self.isVisible() and not self._closing_window:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("自动记录已结束")
            dialog.setText(summary)
            dialog.setDetailedText(paths or "本次没有收到采样数据")
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.finished.connect(lambda: setattr(self, "_record_summary_dialog", None))
            self._record_summary_dialog = dialog
            dialog.show()

    def export_csv(self):
        """将当前显示的八通道原始或滤波数据导出为CSV。"""
        if self._export_worker is not None:
            return
        data_source = self.history_data if self.is_history_mode else self._current_data_source()
        if not any(len(channel) > 0 for channel in data_source):
            QMessageBox.information(self, "提示", "数据缓存区为空，请先采集数据！")
            return

        now = datetime.datetime.now()
        data_tag = "history" if self.is_history_mode else "filtered" if self.filter_enabled else "raw"
        sample_rate = self.history_rate if self.is_history_mode else self.AdsData.sample_rate
        default_name = (
            (subject_filename_prefix(self.collection_subject, self.collection_started_at)
             if not self.is_history_mode and self.collection_subject else
             now.strftime("%Y%m%d_%H%M%S"))
            + "_"
            + str(sample_rate)
            + "sps_"
            + data_tag
            + ".csv"
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

        self._export_data_name = "历史数据" if self.is_history_mode else "滤波数据" if self.filter_enabled else "原始数据"
        self.btn_export.setEnabled(False)
        self._export_progress = QProgressDialog("正在导出 CSV…", "取消", 0, 100, self)
        self._export_progress.setWindowTitle("导出 CSV")
        self._export_progress.setWindowModality(Qt.NonModal)
        self._export_progress.setMinimumDuration(0)
        self._export_progress.setAutoClose(False)
        self._export_progress.setAutoReset(False)
        self._export_worker = CsvExportWorker(filepath, data_source, self.channel_names, self)
        self._export_progress.canceled.connect(self._export_worker.requestInterruption)
        self._export_worker.progress.connect(self._export_progress.setValue)
        self._export_worker.finished.connect(self._finish_csv_export)
        self._export_progress.show()
        self._export_worker.start()

    def _finish_csv_export(self):
        worker = self._export_worker
        if worker is None:
            return
        self._export_worker = None
        self._export_progress.close()
        self._export_progress.deleteLater()
        self._export_progress = None
        self.btn_export.setEnabled(True)
        if not self._closing_window:
            if worker.error:
                QMessageBox.critical(self, "导出失败", f"CSV文件写入失败：\n{worker.error}")
            elif worker.cancelled:
                self.statusBar().showMessage("导出已取消，未完成文件已删除", 5000)
            else:
                QMessageBox.information(self, "提示", f"{self._export_data_name}导出成功！")
        worker.deleteLater()

    def closeEvent(self, event):
        """关闭窗口前停止采集、让设备休眠并释放串口。"""
        self._closing_window = True
        if self._export_worker is not None:
            self._export_worker.requestInterruption()
            self._export_worker.wait()
            self._finish_csv_export()
        self._shutdown_device()
        was_collecting = self.is_collecting
        self.is_collecting = False
        self.is_impedance_testing = False
        self.stop_impedance_test()
        self.stop_collection()
        if was_collecting:
            self._reset_subject_for_next_session()
        self.stats_timer.stop()
        self.ready_timer.stop()
        if self.alpha_model_dialog is not None:
            self.alpha_model_dialog.close()
        if self.motor_imagery_dialog is not None:
            self.motor_imagery_dialog.close()
        if self.rhythm_analysis_dialog is not None:
            self.rhythm_analysis_dialog.close()
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
