"""Personal left/right hand motor-imagery training and live model control."""

from collections import deque
import json
import math
import random
import time

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from motor_imagery import MotorImageryClassifier


class MotorImageryScene(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.position = 0.0
        self.active = False
        self.setMinimumHeight(155)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0B1D2E"))
        left, right = 90.0, self.width() - 90.0
        center_y = self.height() * 0.42
        painter.setPen(QPen(QColor("#355574"), 3))
        painter.drawLine(QPointF(left, center_y), QPointF(right, center_y))
        painter.setFont(QFont("Microsoft YaHei", 11))
        for ratio, text in ((0.0, "左手想象"), (0.5, "中心"), (1.0, "右手想象")):
            x = left + (right - left) * ratio
            painter.setPen(QPen(QColor("#7191AD"), 1))
            painter.drawLine(QPointF(x, center_y - 12), QPointF(x, center_y + 12))
            painter.setPen(QColor("#B6CADC"))
            painter.drawText(QRectF(x - 75, center_y + 38, 150, 28), Qt.AlignCenter, text)
        x = (left + right) / 2.0 + self.position * (right - left) / 2.0
        color = QColor("#47DCC5" if self.active else "#7191AD")
        painter.setPen(QPen(color.lighter(125), 2))
        painter.setBrush(color)
        painter.drawEllipse(QPointF(x, center_y), 23, 23)


class MotorImageryDialog(QDialog):
    STREAM_TIMEOUT_SECONDS = 1.5
    ANALYSIS_SECONDS = 0.25
    WINDOW_SECONDS = 2.0
    TRIALS_PER_CLASS = 10
    MAX_ATTEMPTS = 40
    PHASE_SECONDS = {"prepare": 1.0, "imagine": 3.0, "rest": 2.0}

    def __init__(self, channel_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("脑电控制 · 左右手运动想象")
        self.resize(930, 730)
        self.setMinimumSize(820, 650)
        self.model = None
        self.sample_count = 0
        self._rate = None
        self._available = [True] * 8
        self._stream_active = False
        self._last_sample_time = None
        self._last_tick_time = time.monotonic()
        self._last_analysis_count = 0
        self._last_decision_time = None
        self._buffers = [deque() for _ in range(8)]
        self._stale = False
        self._training = False
        self._controlling = False
        self._phase = None
        self._phase_started = None
        self._phase_samples = 0
        self._trial_buffer = deque()
        self._trials = []
        self._labels = []
        self._queue = []
        self._current_label = None
        self._attempts = 0
        self._votes = deque(maxlen=3)
        self._command = 0
        self._strength = None
        self._message = "请开始 EEG 采集，并选择实际接在 C3、C4 的输入通道。"

        self.setStyleSheet("""
            QDialog, QWidget { background: #071426; color: #DDE7F2;
                font-family: 'Microsoft YaHei'; font-size: 14px; }
            QPushButton, QComboBox { background: #10243B; border: 1px solid #355574;
                border-radius: 5px; padding: 6px 10px; min-height: 24px; }
            QPushButton:hover { background: #173451; }
            QPushButton:disabled { color: #647B91; border-color: #25394D; }
            QComboBox QAbstractItemView { background: #10243B; selection-background-color: #245579; }
            QProgressBar { border: 1px solid #355574; border-radius: 4px;
                background: #0B1D2E; text-align: center; min-height: 20px; }
            QProgressBar::chunk { background: #228E7D; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        title = QLabel("左右手运动想象训练", self)
        title.setStyleSheet("font-size: 23px; font-weight: bold;")
        layout.addWidget(title)
        instructions = QLabel(
            "将电极贴在 C3（头顶左侧）、C4（头顶右侧），选择对应硬件通道。\n"
            "按提示想象左手或右手握拳的感觉，双手保持不动、眼睛睁开。", self)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        source_row = QHBoxLayout()
        self.c3_combo, self.c4_combo = QComboBox(self), QComboBox(self)
        for caption, combo in (("C3 输入", self.c3_combo), ("C4 输入", self.c4_combo)):
            source_row.addWidget(QLabel(caption, self))
            combo.addItem("请选择实际通道", -1)
            for index, name in enumerate(channel_names[:8]):
                combo.addItem(f"CH{index + 1}（主界面 {name}）", index)
            source_row.addWidget(combo, 1)
            combo.currentIndexChanged.connect(self._channels_changed)
        layout.addLayout(source_row)

        self.cue_label = QLabel("等待个人训练", self)
        self.cue_label.setAlignment(Qt.AlignCenter)
        self.cue_label.setMinimumHeight(62)
        self.cue_label.setStyleSheet("background: #10243B; border-radius: 6px; font-size: 26px; font-weight: bold;")
        layout.addWidget(self.cue_label)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, self.TRIALS_PER_CLASS * 2)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("有效训练试次 %v / %m")
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel(self._message, self)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(42)
        layout.addWidget(self.status_label)
        self.scene = MotorImageryScene(self)
        layout.addWidget(self.scene, 1)

        metrics = QGridLayout()
        self.validation_label = QLabel("未训练", self)
        self.direction_label = QLabel("等待", self)
        self.strength_label = QLabel("--", self)
        for column, (caption, label) in enumerate((
            ("试次交叉验证准确率", self.validation_label),
            ("当前判别", self.direction_label),
            ("判别强度", self.strength_label),
        )):
            heading = QLabel(caption, self)
            heading.setAlignment(Qt.AlignCenter)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 21px; font-weight: bold;")
            metrics.addWidget(heading, 0, column)
            metrics.addWidget(label, 1, column)
            metrics.setColumnStretch(column, 1)
        self.strength_label.setToolTip("分类器对本次窗口的判别强度，不能作为真实控制准确率。")
        layout.addLayout(metrics)

        buttons = QHBoxLayout()
        self.train_button = QPushButton("开始训练（约 2 分钟）", self)
        self.train_button.clicked.connect(self.toggle_training)
        self.control_button = QPushButton("开始控制", self)
        self.control_button.clicked.connect(self.toggle_control)
        self.center_button = QPushButton("小球回中", self)
        self.center_button.clicked.connect(self.center_model)
        self.save_button = QPushButton("保存模型", self)
        self.save_button.clicked.connect(self.save_model)
        self.load_button = QPushButton("载入模型", self)
        self.load_button.clicked.connect(self.load_model)
        for button in (self.train_button, self.control_button, self.center_button, self.save_button, self.load_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        hint = QLabel(
            "训练验证通过后开放控制（准确率至少 70%）；交叉验证结果不等于实际控制准确率。\n"
            "只区分左手与右手想象，不控制时请暂停；模型用于本人相同电极位置，换位置后重新训练。", self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #A9BDCF; font-size: 12px;")
        layout.addWidget(hint)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._update_readouts()

    def _channels(self):
        channels = (self.c3_combo.currentData(), self.c4_combo.currentData())
        if any(not isinstance(index, int) or index < 0 for index in channels):
            return None
        if channels[0] == channels[1] or not all(self._available[index] for index in channels):
            return None
        return channels

    def _clear_window(self):
        for buffer in self._buffers:
            buffer.clear()
        self._last_analysis_count = self.sample_count
        self._votes.clear()
        self._command = 0
        self._strength = None
        self._last_decision_time = None
        self.scene.active = False

    def _abort_training(self):
        self._training = False
        self._phase = None
        self._trial_buffer.clear()
        self._queue = []
        self._current_label = None

    def _channels_changed(self):
        self._abort_training()
        self._controlling = False
        self.model = None
        self._trials, self._labels = [], []
        self._clear_window()
        self._message = "通道已改变，请按实际 C3、C4 位置选择不同通道，然后训练。"
        self._update_readouts()

    def set_available_channels(self, states):
        before = self._channels()
        self._available = (list(bool(value) for value in states) + [False] * 8)[:8]
        for combo in (self.c3_combo, self.c4_combo):
            for index in range(8):
                item = combo.model().item(index + 1)
                if item is not None:
                    item.setEnabled(self._available[index])
        if before != self._channels():
            self._channels_changed()
        else:
            self._update_readouts()

    def start_session(self):
        self._abort_training()
        self._controlling = False
        self.model = None
        self._trials, self._labels = [], []
        self.sample_count = 0
        self._rate = None
        self._last_sample_time = None
        self._stale = False
        self._clear_window()
        self.scene.position = 0.0
        self._message = "新采集：请确认 C3、C4 通道，训练或载入本人模型。"
        self._update_readouts()

    def set_stream_active(self, active, reason=""):
        if not active or bool(active) != self._stream_active:
            self._abort_training()
            self._controlling = False
            self._clear_window()
            self._last_sample_time = None
            self._stale = False
        self._stream_active = bool(active)
        if reason:
            self._message = reason
        elif active:
            self._message = "请选择 C3、C4 的实际输入通道，开始个人训练。"
        else:
            self._message = "采集已停止，训练和控制已停止。"
        self._update_readouts()

    def toggle_training(self):
        if self._training:
            self._abort_training()
            self._message = "训练已停止。再次开始会重新采集全部试次。"
            self._update_readouts()
            return
        if not self._stream_active or self._stale or self._rate is None or self._channels() is None:
            return
        self._controlling = False
        self._clear_window()
        self.model = None
        self._trials, self._labels = [], []
        self._queue = [-1] * self.TRIALS_PER_CLASS + [1] * self.TRIALS_PER_CLASS
        random.SystemRandom().shuffle(self._queue)
        self._attempts = 0
        self._training = True
        self.scene.position = 0.0
        self._next_trial()

    def _next_trial(self):
        if not self._queue:
            self._finish_training()
            return
        if self._attempts >= self.MAX_ATTEMPTS:
            self._abort_training()
            self._message = "无效试次过多，训练已停止。请检查电极并保持静止后重试。"
            self._update_readouts()
            return
        self._current_label = self._queue.pop(0)
        self._attempts += 1
        self._set_phase("prepare")

    def _set_phase(self, phase):
        self._phase = phase
        self._phase_started = time.monotonic()
        self._phase_samples = 0
        self._trial_buffer = deque(maxlen=round(self.WINDOW_SECONDS * self._rate))
        self._update_readouts()

    def _advance_training(self, values):
        self._phase_samples += len(values)
        if self._phase == "imagine":
            self._trial_buffer.extend(values)
        duration = self.PHASE_SECONDS[self._phase]
        # Both elapsed time and fresh EEG samples must cover the visible cue.
        # Never assign one buffered packet to multiple unseen prompts.
        if (time.monotonic() - self._phase_started < duration
                or self._phase_samples < round(duration * self._rate)):
            return
        if self._phase == "prepare":
            self._set_phase("imagine")
        elif self._phase == "imagine":
            window_size = round(self.WINDOW_SECONDS * self._rate)
            trial = np.asarray(self._trial_buffer, dtype=np.float64)[-window_size:].copy()
            try:
                MotorImageryClassifier.validate_epoch(trial, self._rate, self._channels())
            except (ValueError, TypeError, FloatingPointError):
                self._queue.append(self._current_label)
                self._message = "本次信号无效，稍后补做；请保持静止并检查电极接触。"
            else:
                self._trials.append(trial)
                self._labels.append(self._current_label)
                self._message = "本次已记录。"
            self._set_phase("rest")
        else:
            self._next_trial()

    def _finish_training(self):
        self._training = False
        self._phase = None
        self._clear_window()
        try:
            model = MotorImageryClassifier()
            model.fit(self._trials, ["left" if value < 0 else "right" for value in self._labels],
                      self._rate, channel_indices=self._channels())
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            self.model = None
            self._message = f"训练未完成：{exc}"
        else:
            self.model = model
            self._message = ("训练验证通过，需实际试控。点击“开始控制”，分别想象左手或右手动作。" if model.ready else
                             "验证未通过，暂不开放控制。需准确率至少 70% 且左右两类均能区分；请检查电极并重新训练。")
        self._update_readouts()

    def toggle_control(self):
        if self._controlling:
            self._controlling = False
            self._message = "控制已暂停。"
        elif (self._stream_active and not self._stale and self._channels() is not None
              and self.model is not None and self.model.ready and self._rate == self.model.sample_rate):
            self._controlling = True
            self._message = "控制中：想象左手动作向左，想象右手动作向右；不确定时停止。"
        self._clear_window()
        self._update_readouts()

    def center_model(self):
        self.scene.position = 0.0
        self.scene.update()

    def feed_samples(self, samples, rate):
        if not self._stream_active or not self.isVisible() or len(samples) == 0:
            return
        try:
            rate = float(rate)
            values = np.asarray(samples, dtype=np.float64)
        except (ValueError, TypeError):
            return
        if not math.isfinite(rate) or not 80 <= rate <= 4000 or values.ndim != 2 or values.shape[1] < 8:
            return
        if self._rate is not None and rate != self._rate:
            self.start_session()
            self._message = "采样率已改变，请重新训练或载入匹配模型。"
        self._check_stream_timeout()
        self._rate = rate
        self._last_sample_time = time.monotonic()
        self._stale = False
        self.sample_count += len(values)
        # A long buffered delivery cannot be reliably aligned with a visual cue.
        # Require fresh short batches again before starting another trial/control.
        if len(values) > round(rate * 0.75):
            self._abort_training()
            self._controlling = False
            self._clear_window()
            self._message = "收到积压数据，训练和控制已停止；数据恢复实时后请重新开始。"
            self._update_readouts()
            return
        if self.model is not None and self.model.sample_rate != rate:
            self.model = None
            self._controlling = False
            self._message = "模型采样率与采集不同，请重新训练或载入匹配模型。"
        window_size = round(self.WINDOW_SECONDS * rate)
        for index, buffer in enumerate(self._buffers):
            if buffer.maxlen != window_size:
                self._buffers[index] = deque(buffer, maxlen=window_size)
            self._buffers[index].extend(values[-window_size:, index])
        if self._training:
            self._advance_training(values[:, :8])
        elif (self._controlling and self.sample_count - self._last_analysis_count >= round(rate * self.ANALYSIS_SECONDS)
              and len(self._buffers[0]) >= window_size):
            self._last_analysis_count = self.sample_count
            self._classify(np.asarray(self._buffers, dtype=np.float64).T)
        self._update_readouts()

    def _classify(self, values):
        try:
            result = self.model.predict(values)
            direction = -1 if result.label == "left" else 1 if result.label == "right" else 0
            confidence = result.confidence
            if not result.signal_valid:
                self._message = "信号无效或幅值过大，已停止移动；请检查电极并保持静止。"
            else:
                self._message = "控制中：分别想象左手或右手动作；不控制时请暂停。"
        except (ValueError, TypeError, FloatingPointError):
            direction, confidence = 0, 0.0
            self._message = "信号无效或幅值过大，已停止移动；请检查电极并保持静止。"
        self._strength = float(confidence)
        self._last_decision_time = time.monotonic()
        if direction:
            self._votes.append(int(direction))
        else:
            self._votes.clear()
        self._command = (self._votes[0] if len(self._votes) == 3 and len(set(self._votes)) == 1 else 0)

    def _check_stream_timeout(self):
        if (self._stream_active and self._last_sample_time is not None
                and time.monotonic() - self._last_sample_time > self.STREAM_TIMEOUT_SECONDS and not self._stale):
            self._stale = True
            self._abort_training()
            self._controlling = False
            self._clear_window()
            self._message = "信号中断，训练和控制已停止；恢复采集后请重新开始。"
            self._update_readouts()

    def _update_readouts(self):
        configured = self._channels() is not None
        ready = self.model is not None and self.model.ready
        self.train_button.setEnabled(self._training or (self._stream_active and not self._stale and configured and self._rate is not None))
        self.train_button.setText("停止训练" if self._training else "重新训练（约 2 分钟）" if self.model is not None else "开始训练（约 2 分钟）")
        self.control_button.setEnabled(self._controlling or (self._stream_active and not self._stale and ready and not self._training and self._rate == self.model.sample_rate))
        self.control_button.setText("暂停控制" if self._controlling else "开始控制")
        self.save_button.setEnabled(self.model is not None and not self._training)
        self.load_button.setEnabled(configured and not self._training and not self._controlling)
        self.c3_combo.setEnabled(not self._training and not self._controlling)
        self.c4_combo.setEnabled(not self._training and not self._controlling)
        self.progress_bar.setValue(len(self._labels))
        self.validation_label.setText("未训练" if self.model is None else f"{self.model.report.balanced_accuracy * 100:.0f}%")
        self.strength_label.setText("--" if self._strength is None else f"{self._strength * 100:.0f}%")
        self.direction_label.setText("左手" if self._command < 0 else "右手" if self._command > 0 else "不确定" if self._controlling else "等待")
        if self._training:
            side = "左手" if self._current_label == -1 else "右手"
            elapsed = min(time.monotonic() - self._phase_started, self._phase_samples / self._rate)
            remaining = max(0, math.ceil(self.PHASE_SECONDS[self._phase] - elapsed))
            cue = {"prepare": "准备", "imagine": f"想象{side}握拳", "rest": "放松休息"}[self._phase]
            self.cue_label.setText(f"{cue} · {remaining} 秒")
            left_count, right_count = self._labels.count(-1), self._labels.count(1)
            self.status_label.setText(f"已记录：左手 {left_count}/{self.TRIALS_PER_CLASS}，右手 {right_count}/{self.TRIALS_PER_CLASS}。{self._message if self._phase == 'rest' else '请跟随提示，保持双手不动。'}")
        else:
            self.cue_label.setText("左右手想象控制中" if self._controlling else "训练已完成" if ready else "需要个人训练")
            self.status_label.setText(self._message)
        self.scene.active = bool(self._controlling and self._command and not self._stale)
        self.scene.update()

    def _tick(self):
        now = time.monotonic()
        elapsed = max(0.0, min(0.1, now - self._last_tick_time))
        self._last_tick_time = now
        self._check_stream_timeout()
        if self._last_decision_time is not None and now - self._last_decision_time > 0.75:
            self._votes.clear()
            self._command = 0
            self.scene.active = False
        if self.scene.active:
            self.scene.position = float(np.clip(self.scene.position + self._command * elapsed * 0.45, -1.0, 1.0))
        self._update_readouts()

    def save_model(self):
        if self.model is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存个人运动想象模型", "motor_imagery.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(self.model.to_dict(), stream, ensure_ascii=False, indent=2)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def load_model(self):
        if self._channels() is None or self._training or self._controlling:
            return
        path, _ = QFileDialog.getOpenFileName(self, "载入个人运动想象模型", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as stream:
                model = MotorImageryClassifier.from_dict(json.load(stream))
            if tuple(model.channel_indices) != self._channels():
                raise ValueError("模型的 C3、C4 输入通道与当前选择不同。请确认接线或重新训练。")
            if model.channel_count != 8:
                raise ValueError("模型不属于当前八通道采集布局，请重新训练。")
            if self._rate is not None and model.sample_rate != self._rate:
                raise ValueError("模型采样率与当前采集不同，请重新训练。")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "载入失败", str(exc))
            return
        self.model = model
        self._clear_window()
        self._message = "个人模型已载入，请确认本人的电极位置与训练时相同。" if model.ready else "模型未通过训练验证，请重新训练。"
        self._update_readouts()

    def showEvent(self, event):
        self._last_tick_time = time.monotonic()
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        self.set_stream_active(False, "窗口已关闭，训练和控制已停止。")
        super().hideEvent(event)
