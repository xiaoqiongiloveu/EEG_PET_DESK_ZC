"""A live Qt model controlled by calibrated EEG band power."""

from collections import deque
import math
import time

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QGridLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from alpha_model_control import AlphaModelController


class AlphaModelScene(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.position = 0.0
        self.active = False
        self.setMinimumHeight(210)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0B1D2E"))
        left, right = 64.0, self.width() - 64.0
        center_y = self.height() * 0.5
        center_x = (left + right) * 0.5
        painter.setPen(QPen(QColor("#355574"), 3))
        painter.drawLine(QPointF(left, center_y), QPointF(right, center_y))
        painter.setFont(QFont("Microsoft YaHei", 10))
        for ratio, text in ((0.0, "-6 dB"), (0.5, "基线 0 dB"), (1.0, "+6 dB")):
            x = left + (right - left) * ratio
            painter.setPen(QPen(QColor("#7191AD"), 1))
            painter.drawLine(QPointF(x, center_y - 12), QPointF(x, center_y + 12))
            painter.setPen(QColor("#B6CADC"))
            painter.drawText(QRectF(x - 60, center_y + 48, 120, 28), Qt.AlignCenter, text)
        painter.setPen(QPen(QColor("#355574"), 1, Qt.DashLine))
        painter.drawLine(QPointF(center_x, 26), QPointF(center_x, center_y - 35))
        x = center_x + self.position * (right - left) * 0.5
        color = QColor("#47DCC5" if self.active else "#7191AD")
        painter.setPen(QPen(color.lighter(125), 2))
        painter.setBrush(color)
        painter.drawEllipse(QPointF(x, center_y), 25, 25)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#E6FFF9"))
        painter.drawEllipse(QPointF(x - 7, center_y - 8), 5, 5)


class AlphaModelDialog(QDialog):
    STREAM_TIMEOUT_SECONDS = 1.5
    ANALYSIS_SECONDS = 0.2

    def __init__(self, channel_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("脑电控制 · 9–13 Hz")
        self.resize(860, 570)
        self.setMinimumSize(720, 510)
        self.controller = AlphaModelController()
        self.sample_count = 0
        self._rate = None
        self._buffers = [deque() for _ in range(8)]
        self._last_analysis_count = 0
        self._last_sample_time = None
        self._last_tick_time = time.monotonic()
        self._stream_active = False
        self._stream_reason = "请先开始 EEG 采集"
        self._stale = False

        self.setStyleSheet("""
            QDialog, QWidget { background: #071426; color: #DDE7F2;
                font-family: 'Microsoft YaHei'; font-size: 14px; }
            QPushButton { background: #10243B; border: 1px solid #355574;
                border-radius: 6px; padding: 7px 14px; min-height: 25px; }
            QPushButton:hover { background: #173451; }
            QPushButton:disabled { color: #647B91; border-color: #25394D; }
            QProgressBar { border: 1px solid #355574; border-radius: 4px;
                background: #0B1D2E; text-align: center; min-height: 20px; }
            QProgressBar::chunk { background: #228E7D; }
            QCheckBox { spacing: 5px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        title = QLabel("9–13 Hz 脑电模型", self)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        self.status_label = QLabel(self._stream_reason, self)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(24)
        layout.addWidget(self.status_label)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("控制通道", self))
        self.channel_checks = []
        for index, name in enumerate(channel_names[:8]):
            check = QCheckBox(name, self)
            check.setChecked(index in (1, 2, 3))
            check.toggled.connect(self._channels_changed)
            channel_row.addWidget(check)
            self.channel_checks.append(check)
        channel_row.addStretch(1)
        layout.addLayout(channel_row)

        self.scene = AlphaModelScene(self)
        layout.addWidget(self.scene, 1)
        self.mapping_label = QLabel("低于基线：向左    |    高于基线：向右    |    非左右意念识别", self)
        self.mapping_label.setWordWrap(True)
        self.mapping_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mapping_label)

        metrics = QGridLayout()
        self.power_label = QLabel("-- dB", self)
        self.baseline_label = QLabel("-- dB", self)
        self.offset_label = QLabel("-- dB", self)
        for column, (name, label) in enumerate((
            ("9–13 Hz 功率", self.power_label),
            ("个人基线", self.baseline_label),
            ("相对变化", self.offset_label),
        )):
            heading = QLabel(name, self)
            heading.setAlignment(Qt.AlignCenter)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 22px; font-weight: bold;")
            metrics.addWidget(heading, 0, column)
            metrics.addWidget(label, 1, column)
            metrics.setColumnStretch(column, 1)
        self.power_label.setToolTip("9–13 Hz 积分功率，dB 相对于 1 µV²；与主界面频谱最高点不同")
        layout.addLayout(metrics)
        self.calibration_bar = QProgressBar(self)
        self.calibration_bar.setRange(0, 100)
        self.calibration_bar.setValue(0)
        self.calibration_bar.setFormat("基线校准 %p%")
        layout.addWidget(self.calibration_bar)

        buttons = QHBoxLayout()
        self.calibrate_button = QPushButton("校准基线（5 秒）", self)
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.pause_button = QPushButton("暂停", self)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.reset_button = QPushButton("重置", self)
        self.reset_button.clicked.connect(self.reset_control)
        buttons.addWidget(self.calibrate_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        hint = QLabel("校准时保持静止；眨眼、肌肉或线缆动作可能影响结果。", self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #A9BDCF; font-size: 13px;")
        layout.addWidget(hint)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._update_readouts()

    def _enabled_channels(self):
        return [check.isEnabled() and check.isChecked() for check in self.channel_checks]

    def _clear_window(self):
        for buffer in self._buffers:
            buffer.clear()
        self._last_analysis_count = self.sample_count

    def _channels_changed(self):
        self.reset_control()

    def set_available_channels(self, states):
        before = self._enabled_channels()
        for check, available in zip(self.channel_checks, states):
            check.setEnabled(bool(available))
        if before != self._enabled_channels():
            self.reset_control()

    def start_session(self):
        self.sample_count = 0
        self._rate = None
        self._last_sample_time = None
        self._stale = False
        self.reset_control()

    def set_stream_active(self, active, reason=""):
        active = bool(active)
        if active != self._stream_active:
            self._clear_window()
            self._last_sample_time = None
            self._stale = False
            if active:
                self.controller.resume()
            else:
                self.controller.pause()
        self._stream_active = active
        self._stream_reason = reason or ("" if active else "采集已停止")
        self.scene.active = False
        self._update_readouts()

    def start_calibration(self):
        if not self._stream_active or not any(self._enabled_channels()):
            return
        self.controller.start_calibration()
        self._clear_window()
        self.scene.position = 0.0
        self.scene.active = False
        self._update_readouts()

    def toggle_pause(self):
        if not self._stream_active:
            return
        if self.controller.status == "paused":
            self.controller.resume()
        else:
            self.controller.pause()
        self._clear_window()
        self._update_readouts()

    def reset_control(self):
        self.controller.reset()
        self._clear_window()
        self.scene.position = 0.0
        self.scene.active = False
        self._update_readouts()

    def feed_samples(self, samples, rate):
        if not self._stream_active or not self.isVisible() or len(samples) == 0:
            return
        rate = float(rate)
        if not math.isfinite(rate) or rate <= 26 or rate > 4000:
            return
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 8:
            return
        if self._rate is not None and rate != self._rate:
            self.sample_count = 0
            self.reset_control()
        self._rate = rate
        self._check_stream_timeout()
        self._last_sample_time = time.monotonic()
        self._stale = False
        self.sample_count += len(values)
        window_size = int(round(rate * self.controller.WINDOW_SECONDS))
        for index in range(8):
            if self._buffers[index].maxlen != window_size:
                self._buffers[index] = deque(self._buffers[index], maxlen=window_size)
            self._buffers[index].extend(values[-window_size:, index])
        if self.sample_count - self._last_analysis_count >= max(1, round(rate * self.ANALYSIS_SECONDS)):
            self.controller.update(self._buffers, rate, self.sample_count, self._enabled_channels())
            self._last_analysis_count = self.sample_count
        self._update_readouts()

    def _check_stream_timeout(self):
        if self._stream_active and self._last_sample_time is not None:
            if time.monotonic() - self._last_sample_time > self.STREAM_TIMEOUT_SECONDS and not self._stale:
                self._stale = True
                # Break the sample-time anchor as well as the raw window, so
                # calibration never credits an unobserved interval on recovery.
                was_paused = self.controller.status == "paused"
                self.controller.pause()
                if not was_paused:
                    self.controller.resume()
                self._clear_window()
                self.scene.active = False
                self._update_readouts()

    def _update_readouts(self):
        controller = self.controller
        self.power_label.setText("-- dB" if controller.band_db is None else f"{controller.band_db:.1f} dB")
        self.baseline_label.setText("-- dB" if controller.baseline_db is None else f"{controller.baseline_db:.1f} dB")
        self.offset_label.setText("-- dB" if controller.offset_db is None else f"{controller.offset_db:+.1f} dB")
        self.calibration_bar.setValue(round(controller.calibration_progress * 100))
        self.calibrate_button.setEnabled(self._stream_active and any(self._enabled_channels()))
        self.pause_button.setEnabled(self._stream_active and controller.status != "waiting")
        self.pause_button.setText("继续" if controller.status == "paused" else "暂停")
        self.calibrate_button.setText("重新校准（5 秒）" if controller.baseline_db is not None else "校准基线（5 秒）")
        ready = self._rate is not None and len(self._buffers[0]) >= round(self._rate * controller.WINDOW_SECONDS)
        if not self._stream_active:
            status = self._stream_reason
        elif not any(self._enabled_channels()):
            status = "请选择至少一个控制通道"
        elif self._stale:
            status = "等待脑电数据：信号中断，模型已冻结"
        elif controller.status == "paused":
            status = "已暂停"
        elif self._last_sample_time is None:
            status = "等待实时 EEG 数据"
        elif not ready:
            status = "正在收集 2 秒脑电分析窗口"
        elif not controller.signal_valid:
            status = "信号无效或幅值过大：请检查电极接触并保持静止"
        elif controller.status == "calibrating":
            status = f"基线校准中：{controller.calibration_progress * 100:.0f}% · 请保持静止"
        elif controller.status == "running":
            status = "实时控制中"
        else:
            status = "脑电数据已就绪，请点击“校准基线”"
        self.status_label.setText(status)
        self.scene.active = bool(self._stream_active and not self._stale and ready
                                 and controller.signal_valid and controller.status == "running")
        self.scene.update()

    def _tick(self):
        now = time.monotonic()
        elapsed = max(0.0, min(0.1, now - self._last_tick_time))
        self._last_tick_time = now
        self._check_stream_timeout()
        if self.scene.active:
            self.scene.position += (self.controller.position - self.scene.position) * (-math.expm1(-elapsed / 0.15))
            self.scene.update()

    def showEvent(self, event):
        self._last_tick_time = time.monotonic()
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        self.set_stream_active(False, "模型窗口已关闭")
        super().hideEvent(event)
