"""Qt view for offline alpha and mu rhythm analysis."""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from history_replay import sample_rate_from_filename
from rhythm_analysis import BAND, analyze_rhythms, read_analysis_csv


def _number(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}"


class RhythmAnalysisDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("EEG 节律分析")
        self.resize(1150, 790)
        self.setMinimumSize(880, 620)
        self._data = None
        self._names = list(parent.channel_names)
        self._source = ""
        self._linked_to_main = False
        self.result = None
        self.setStyleSheet("""
            QDialog, QWidget { background: #071426; color: #DDE7F2;
                font-family: 'Microsoft YaHei'; font-size: 13px; }
            QPushButton, QComboBox, QDoubleSpinBox, QLineEdit {
                background: #10243B; color: #EAF2FA; border: 1px solid #355574;
                border-radius: 5px; min-height: 28px; padding: 2px 7px; }
            QPushButton:hover { background: #173451; }
            QPushButton:disabled { color: #72869A; }
            QTabWidget::pane { border: 1px solid #2A425E; }
            QTabBar::tab { background: #10243B; padding: 8px 18px; }
            QTabBar::tab:selected { background: #24517A; }
            QTableWidget, QPlainTextEdit { background: #0B1D2E; color: #DDE7F2;
                gridline-color: #284057; border: 1px solid #2A425E; }
            QTableWidget { alternate-background-color: #10243B;
                selection-background-color: #24517A; }
            QHeaderView::section { background: #173451; color: #EAF2FA;
                border: 0; border-right: 1px solid #2A425E; padding: 5px; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        source_row = QHBoxLayout()
        self.current_button = QPushButton("读取当前数据", self)
        self.current_button.clicked.connect(self.load_current_data)
        self.csv_button = QPushButton("打开 CSV", self)
        self.csv_button.clicked.connect(self.load_csv)
        self.source_label = QLabel("未加载数据", self)
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_row.addWidget(self.current_button)
        source_row.addWidget(self.csv_button)
        source_row.addWidget(self.source_label, 1)
        layout.addLayout(source_row)

        options = QHBoxLayout()
        self.rate_spin = QDoubleSpinBox(self)
        self.rate_spin.setRange(80, 4000)
        self.rate_spin.setDecimals(0)
        self.rate_spin.setValue(250)
        self.rate_spin.setSuffix(" Hz")
        self.threshold_spin = QDoubleSpinBox(self)
        self.threshold_spin.setRange(10, 5000)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(200)
        self.threshold_spin.setSuffix(" µV")
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("α + μ", "both")
        self.mode_combo.addItem("仅 α", "alpha")
        self.mode_combo.addItem("仅 μ", "mu")
        self.hand_combo = QComboBox(self)
        self.hand_combo.addItem("右手", "right")
        self.hand_combo.addItem("左手", "left")
        self.hand_combo.addItem("双手/未知", "both")
        for label, control in (("采样率", self.rate_spin), ("伪迹阈值", self.threshold_spin),
                               ("分析目标", self.mode_combo), ("运动侧", self.hand_combo)):
            options.addWidget(QLabel(label, self))
            options.addWidget(control)
        options.addStretch(1)
        layout.addLayout(options)

        segment_grid = QGridLayout()
        segment_grid.setHorizontalSpacing(12)
        segment_grid.setVerticalSpacing(6)
        self.segment_edits = {}
        fields = (
            ("closed", "闭眼", 0, 0), ("open", "睁眼", 0, 1),
            ("rest", "静息", 1, 0), ("move", "运动/想象", 1, 1),
            ("post", "运动后", 2, 0), ("eyesopen", "睁眼静息", 2, 1),
        )
        for key, title, row, column in fields:
            edit = QLineEdit(self)
            edit.setPlaceholderText("秒，如 0-20,100-120")
            edit.setToolTip(f"{title}时间段，单位秒；多段以逗号分隔")
            segment_grid.addWidget(QLabel(title, self), row, column * 2)
            segment_grid.addWidget(edit, row, column * 2 + 1)
            self.segment_edits[key] = edit
        segment_grid.setColumnStretch(1, 1)
        segment_grid.setColumnStretch(3, 1)
        layout.addLayout(segment_grid)

        actions = QHBoxLayout()
        self.analyze_button = QPushButton("开始分析", self)
        self.analyze_button.clicked.connect(self.run_analysis)
        self.export_button = QPushButton("导出报告", self)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_report)
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #A9BDCF;")
        actions.addWidget(self.analyze_button)
        actions.addWidget(self.export_button)
        actions.addWidget(self.status_label, 1)
        layout.addLayout(actions)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)
        overview = QWidget(self)
        overview_layout = QVBoxLayout(overview)
        self.summary_label = QLabel("", overview)
        overview_layout.addWidget(self.summary_label)
        self.quality_figure, self.quality_canvas = self._figure()
        overview_layout.addWidget(self.quality_canvas, 1)
        self.notes_box = QPlainTextEdit(overview)
        self.notes_box.setReadOnly(True)
        self.notes_box.setMaximumHeight(100)
        overview_layout.addWidget(self.notes_box)
        self.tabs.addTab(overview, "概况")

        self.spectrum_figure, self.spectrum_canvas = self._figure()
        self.spectrum_table = self._table(("通道", "峰值 Hz", "显著性", "谐波 Hz", "8–13 Hz占比 %", "弧状指数"))
        self.tabs.addTab(self._chart_tab(self.spectrum_canvas, self.spectrum_table), "频谱")
        self.alpha_figure, self.alpha_canvas = self._figure()
        self.alpha_table = self._table(("通道", "IAF Hz", "显著性", "带宽 Hz", "占比 %",
                                        "纺锤数", "个/分钟", "平均时长 s", "平均幅度 µV",
                                        "睁眼抑制 %", "Berger", "判定"))
        self.tabs.addTab(self._chart_tab(self.alpha_canvas, self.alpha_table), "α 节律")
        self.mu_figure, self.mu_canvas = self._figure()
        self.mu_table = self._table(("通道", "静息功率", "运动功率", "ERD %", "β 回弹 %",
                                     "睁眼抑制 %", "弧状指数", "判定", "μ/α"))
        self.tabs.addTab(self._chart_tab(self.mu_canvas, self.mu_table), "μ 节律")

        self.rate_spin.valueChanged.connect(self._invalidate_result)
        self.threshold_spin.valueChanged.connect(self._invalidate_result)
        self.mode_combo.currentIndexChanged.connect(self._invalidate_result)
        self.hand_combo.currentIndexChanged.connect(self._invalidate_result)
        for edit in self.segment_edits.values():
            edit.textEdited.connect(self._invalidate_result)

        self.load_current_data(silent=True)

    @staticmethod
    def _figure():
        figure = Figure(figsize=(8, 3), facecolor="#071426")
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(180)
        return figure, canvas

    @staticmethod
    def _table(columns):
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        return table

    @staticmethod
    def _chart_tab(canvas, table):
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(canvas)
        splitter.addWidget(table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        return splitter

    @staticmethod
    def _style_axis(axis):
        axis.set_facecolor("#0B1D2E")
        axis.tick_params(colors="#C4CFDB", labelsize=9)
        axis.xaxis.label.set_color("#C4CFDB")
        axis.yaxis.label.set_color("#C4CFDB")
        axis.title.set_color("#EAF2FA")
        axis.grid(color="#355574", alpha=0.45)
        for spine in axis.spines.values():
            spine.set_color("#50647A")

    def _set_source(self, data, rate, names, caption, linked_to_main=False):
        self._invalidate_result()
        self._data = np.asarray(data, dtype=np.float64)
        self._names = list(names)
        self._source = caption
        self._linked_to_main = linked_to_main
        self.rate_spin.setValue(rate)
        self.source_label.setText(f"{caption} · {self._data.shape[1] / rate:.1f} 秒")
        self.source_label.setToolTip(caption)
        self.status_label.setText("数据已加载")

    def _invalidate_result(self):
        if self.result is None:
            return
        self.result = None
        self.export_button.setEnabled(False)
        self.status_label.setText("参数已更新，请重新分析")
        self.summary_label.clear()
        self.notes_box.clear()
        for table in (self.spectrum_table, self.alpha_table, self.mu_table):
            table.setRowCount(0)
        for figure, canvas in ((self.quality_figure, self.quality_canvas),
                               (self.spectrum_figure, self.spectrum_canvas),
                               (self.alpha_figure, self.alpha_canvas),
                               (self.mu_figure, self.mu_canvas)):
            figure.clear()
            canvas.draw()

    def load_current_data(self, silent=False):
        parent = self.parent()
        if parent.is_history_mode and parent.history_data is not None:
            source = parent.history_data
            rate = parent.history_rate
            caption = f"当前回访：{Path(parent.history_file_path).name}"
        else:
            source = parent._current_data_source()
            rate = int(parent.AdsData.sample_rate or 250)
            caption = "当前采集（滤波）" if parent.filter_enabled else "当前采集（原始）"
        length = min(map(len, source)) if source else 0
        if length == 0:
            if not silent:
                QMessageBox.information(self, "暂无数据", "请先采集脑电、打开历史回访，或载入 CSV。")
            return
        snapshot = np.stack([np.asarray(channel[:length], dtype=np.float64) for channel in source])
        self._set_source(snapshot, rate, parent.channel_names, caption, linked_to_main=True)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 EEG CSV", "", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            parent = self.parent()
            names, channels, linked = read_analysis_csv(
                path, parent.channel_names, parent.channel_name_history,
            )
            rate = sample_rate_from_filename(path) or int(self.rate_spin.value())
            if not 80 <= rate <= 4000:
                raise ValueError("采样率须在 80–4000 Hz 之间")
            self._set_source(channels, rate, names, Path(path).name, linked_to_main=linked)
            if sample_rate_from_filename(path) is None:
                self.status_label.setText("请核对采样率后开始分析")
        except (OSError, UnicodeError, ValueError) as exc:
            QMessageBox.warning(self, "CSV 读取失败", str(exc))

    def rename_channel(self, index, name):
        if not self._linked_to_main:
            return
        old = self._names[index]
        self._names[index] = name
        if self.result is not None:
            self.result["channels"][index] = name
            for key in ("spectrum", "alpha", "mu", "scans"):
                for row in self.result[key]:
                    if row["channel"] == old:
                        row["channel"] = name
            self.result["notes"] = [note.replace(old, name) for note in self.result["notes"]]
            self._render(self.result)

    def run_analysis(self):
        if self._data is None:
            QMessageBox.information(self, "暂无数据", "请先读取当前数据或打开 CSV。")
            return
        self.analyze_button.setEnabled(False)
        self.status_label.setText("正在分析...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = analyze_rhythms(
                self._data, self.rate_spin.value(), self._names,
                bad_threshold=self.threshold_spin.value(),
                mode=self.mode_combo.currentData(),
                segments={key: edit.text() for key, edit in self.segment_edits.items()},
                scan=True, hand=self.hand_combo.currentData(),
            )
            self.result = result
            self._render(result)
            self.export_button.setEnabled(True)
            self.status_label.setText("分析完成")
        except (ValueError, FloatingPointError) as exc:
            self.status_label.setText("分析失败")
            QMessageBox.warning(self, "分析失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.analyze_button.setEnabled(True)

    @staticmethod
    def _fill_table(table, rows):
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()

    def _render(self, result):
        duration = result["duration_s"]
        self.summary_label.setText(
            f"{self._source}    {result['sample_rate']} Hz    {duration:.1f} 秒    "
            f"可用 {result['good_seconds']:.1f} 秒    坏窗 {result['bad_fraction'] * 100:.1f}%"
        )
        self.notes_box.setPlainText("\n".join(result["notes"]) or "已完成频谱和节律计算。")
        self._render_quality(result)
        self._render_spectrum(result)
        self._render_alpha(result)
        self._render_mu(result)

    def _render_quality(self, result):
        figure = self.quality_figure
        figure.clear()
        axis = figure.add_subplot(111)
        x = np.arange(len(result["max_uv"]))
        axis.plot(x, np.maximum(result["max_uv"], 0.01), color="#51B5CF", linewidth=0.9)
        axis.axhline(result["bad_threshold"], color="#E6B652", linestyle="--", linewidth=1)
        axis.fill_between(x, 0.01, np.maximum(result["max_uv"], 0.01),
                          where=result["bad_mask"], color="#E96A67", alpha=0.4)
        axis.set_yscale("log")
        axis.set_xlabel("时间 (秒)")
        axis.set_ylabel("每秒最大偏移 (µV)")
        axis.set_title("伪迹分布")
        self._style_axis(axis)
        figure.tight_layout()
        self.quality_canvas.draw()

    def _render_spectrum(self, result):
        figure = self.spectrum_figure
        figure.clear()
        axis = figure.add_subplot(111)
        frequencies = result["frequencies"]
        colors = ("#4C9FFF", "#FF9B5A", "#F47DBA", "#53D5A1", "#AE91F2",
                  "#61C9E9", "#E3C05A", "#42C7C3")
        for index, name in enumerate(result["channels"]):
            axis.semilogy(frequencies, np.maximum(result["psds"][index], 1e-12),
                          label=name, linewidth=1, color=colors[index % len(colors)])
        axis.axvspan(*BAND, color="#E6B652", alpha=0.18)
        axis.axvspan(20, 30, color="#AE91F2", alpha=0.10)
        axis.set_xlim(0.5, 45)
        axis.set_xlabel("频率 (Hz)")
        axis.set_ylabel("PSD (µV²/Hz)")
        axis.set_title("各通道功率谱 · α/μ 8–13 Hz")
        self._style_axis(axis)
        axis.legend(ncol=4, fontsize=8, frameon=False, labelcolor="#EAF2FA")
        figure.tight_layout()
        self.spectrum_canvas.draw()
        self._fill_table(self.spectrum_table, [
            (row["channel"], _number(row["peak_hz"], 2), _number(row["saliency"], 2),
             _number(row["harmonic_hz"], 1), _number(row["band_ratio_pct"], 1),
             _number(row["arcade_index"], 3)) for row in result["spectrum"]
        ])

    def _render_alpha(self, result):
        rows = result["alpha"]
        self._fill_table(self.alpha_table, [
            (row["channel"], _number(row["iaf_hz"], 2), _number(row["saliency"], 2),
             _number(row["bandwidth_hz"], 2), _number(row["band_ratio_pct"], 1),
             row["spindles"], _number(row["spindles_per_min"], 1),
             _number(row["spindle_duration_s"], 2), _number(row["spindle_amp_uv"], 1),
             _number(row["berger_pct"], 1), row["berger_verdict"] or "--", row["verdict"])
            for row in rows
        ])
        figure = self.alpha_figure
        figure.clear()
        axis = figure.add_subplot(111)
        if rows:
            x = np.arange(len(rows))
            values = [row["band_ratio_pct"] for row in rows]
            axis.bar(x, values, color="#D7B357", width=0.65)
            axis.set_xticks(x, [row["channel"] for row in rows])
            axis.set_ylabel("α 频段功率占比 (%)")
            axis.set_title("α 频段分布")
            if any(row["berger_pct"] is not None for row in rows):
                other = axis.twinx()
                other.plot(x, [row["berger_pct"] if row["berger_pct"] is not None else np.nan
                               for row in rows], color="#53D5A1", marker="o")
                other.set_ylabel("睁眼抑制 (%)", color="#53D5A1")
                other.tick_params(colors="#53D5A1")
        self._style_axis(axis)
        figure.tight_layout()
        self.alpha_canvas.draw()

    def _render_mu(self, result):
        rows = result["mu"]
        self._fill_table(self.mu_table, [
            (row["channel"], _number(row["rest_power"], 2), _number(row["move_power"], 2),
             _number(row["erd_pct"], 1), _number(row["beta_ers_pct"], 1),
             _number(row["eye_suppression_pct"], 1), _number(row["arcade_index"], 3),
             row["verdict"], row["distinction"] or "--") for row in rows
        ])
        figure = self.mu_figure
        figure.clear()
        axis = figure.add_subplot(111)
        if rows:
            x = np.arange(len(rows))
            axis.bar(x, [row["erd_pct"] or 0 for row in rows], color="#5AB7D1")
            axis.axhline(-40, color="#E96A67", linestyle="--", linewidth=1)
            axis.set_xticks(x, [row["channel"] for row in rows])
            axis.set_ylabel("ERD (%)")
            axis.set_title("运动/想象相对静息的 μ 功率变化")
        elif result["scans"]:
            rows = result["scans"]
            x = np.arange(len(rows))
            axis.bar(x, [row["change_pct"] for row in rows], color="#5AB7D1")
            axis.set_xticks(x, [row["channel"] for row in rows])
            axis.set_ylabel("段内最低变化 (%)")
            axis.set_title("无标注 ERD 候选扫描")
            self._fill_table(self.mu_table, [
                (row["channel"], "--", "--", _number(row["change_pct"], 1),
                 "--", "--", "--", f"候选 @ {row['time_s']:.1f} s", "--")
                for row in rows
            ])
        self._style_axis(axis)
        figure.tight_layout()
        self.mu_canvas.draw()

    def export_report(self):
        if self.result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择报告目录", "")
        if not directory:
            return
        prefix = "rhythm_report_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(directory)
        try:
            result = self.result
            for key in ("spectrum", "alpha", "mu", "scans"):
                rows = result[key]
                if not rows:
                    continue
                with (root / f"{prefix}_{key}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with (root / f"{prefix}.txt").open("w", encoding="utf-8") as stream:
                stream.write(self.summary_label.text() + "\n")
                stream.write(f"分析目标: {result['mode']}\n")
                stream.write(f"伪迹阈值: {result['bad_threshold']:g} µV\n")
                stream.write(f"α/μ 分析频段: {BAND[0]:g}–{BAND[1]:g} Hz\n")
                for key, ranges in result["segments"].items():
                    if ranges:
                        stream.write(f"{key}: {ranges}\n")
                stream.write("\n" + self.notes_box.toPlainText() + "\n")
            for name, figure in (("quality", self.quality_figure),
                                 ("spectrum", self.spectrum_figure),
                                 ("alpha", self.alpha_figure), ("mu", self.mu_figure)):
                figure.savefig(root / f"{prefix}_{name}.png", dpi=130, facecolor=figure.get_facecolor())
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.status_label.setText(f"已导出到 {root}")
