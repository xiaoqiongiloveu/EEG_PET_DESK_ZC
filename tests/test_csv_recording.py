"""验证记录边界、回放兼容、磁盘异常隔离与后台导出。"""

import csv
import datetime
import importlib.util
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from ads1299 import ADC_COUNT_TO_UV, BP_TYPE_EEG, bp_pack
from csv_recording import (CsvExportWorker, SegmentedCsvRecorder, segment_filename,
                           subject_filename_prefix)
from history_replay import read_history_csv

CHANNELS = ["F5", "F3", "FZ", "F4", "F6", "AF8", "FPZ", "AF7"]
START = datetime.datetime(2026, 10, 29, 12, 34, 56)


def rows(count, offset=0):
    return [tuple(index + channel / 10 for channel in range(8))
            for index in range(offset, offset + count)]


@pytest.mark.parametrize("seconds,max_rows", [(300, 3), (1, 0)])
def test_segments_round_trip_without_duplicates(tmp_path, seconds, max_rows):
    rate = 3
    recorder = SegmentedCsvRecorder(tmp_path, seconds, max_rows, started_at=START)
    assert recorder.submit(rows(4), rate, "raw", CHANNELS)
    assert recorder.submit(rows(4, 4), rate, "raw", CHANNELS)
    paths = recorder.close()
    assert [Path(path).name for path in paths] == [
        f"20261029_123456_3sps_raw_{sequence:03d}.csv" for sequence in (1, 2, 3)
    ]
    assert segment_filename(START, 250, "filtered", 1000).endswith("filtered_1000.csv")
    loaded = [read_history_csv(path, CHANNELS) for path in paths]
    assert [len(data[0]) for data in loaded] == [3, 3, 2]
    np.testing.assert_allclose(np.concatenate(loaded, axis=1).T, rows(8))
    assert recorder._stream is None
    assert not recorder._thread.is_alive()
    assert recorder.close() == paths
    assert not recorder.submit(rows(1), rate, "raw", CHANNELS)
    for path in paths:
        # Windows 上重命名可检查是否仍有未释放的文件句柄。
        renamed = Path(path).with_suffix(".closed")
        Path(path).rename(renamed)


def test_same_second_restart_never_overwrites(tmp_path):
    paths = []
    for _ in range(2):
        recorder = SegmentedCsvRecorder(tmp_path, started_at=START)
        recorder.submit(rows(1), 250, "raw", CHANNELS)
        paths.extend(recorder.close())
    assert [Path(path).name for path in paths] == [
        "20261029_123456_250sps_raw_001.csv",
        "20261029_123456_250sps_raw_002.csv",
    ]
    assert all(len(read_history_csv(path, CHANNELS)[0]) == 1 for path in paths)


def test_incremental_flush_and_source_change(tmp_path):
    written = threading.Event()
    recorder = SegmentedCsvRecorder(tmp_path, on_path=lambda path: written.set())
    recorder.submit(rows(256), 1000, "raw", CHANNELS)
    assert written.wait(3)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            if len(read_history_csv(recorder.paths[0], CHANNELS)[0]) == 256:
                break
        except ValueError:
            pass
        time.sleep(0.01)
    else:
        pytest.fail("累计 256 点后未增量落盘")
    recorder.submit(rows(2, 256), 1000, "filtered", CHANNELS)
    paths = recorder.close()
    assert "raw_001" in paths[0]
    assert "filtered_002" in paths[1]
    assert len(read_history_csv(paths[1], CHANNELS)[0]) == 2


def test_short_tail_flushes_after_one_second(tmp_path):
    recorder = SegmentedCsvRecorder(tmp_path)
    try:
        recorder.submit(rows(1), 250, "raw", CHANNELS)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if recorder.paths:
                try:
                    if len(read_history_csv(recorder.paths[0], CHANNELS)[0]) == 1:
                        break
                except ValueError:
                    pass
            time.sleep(0.01)
        else:
            pytest.fail("不足一批的数据未在约一秒后写盘")
    finally:
        recorder.close()


@pytest.fixture(scope="module")
def qt_app():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def window(qt_app, monkeypatch):
    spec = importlib.util.spec_from_file_location("recording_main", ROOT / "python_ads1299/main_uart_v1.2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window = module.MainWindow()
    window._test_subject_dialog_class = module.SubjectDialog
    window.update_show_timer.stop()
    window.stats_timer.stop()
    monkeypatch.setattr(window, "_shutdown_device", lambda: None)
    monkeypatch.setattr(window, "update_graph", lambda *args, **kwargs: False)
    monkeypatch.setattr(window, "_update_spectrum", lambda *args: None)
    monkeypatch.setattr(window.canvas, "draw_idle", lambda: None)
    monkeypatch.setattr(window.spectrum_canvas, "draw_idle", lambda: None)
    yield window
    window.close()
    from matplotlib import pyplot as plt
    plt.close(window.fig)
    plt.close(window.spectrum_fig)


class SerialStub:
    is_open = True

    def __init__(self):
        self.payload = b""

    @property
    def in_waiting(self):
        return len(self.payload)

    def read(self, size):
        result, self.payload = self.payload[:size], self.payload[size:]
        return result

    def write(self, data):
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.is_open = False


def start(window):
    window.serial_port = SerialStub()
    window.subject_info = {"kind": "person", "name": "张三", "age": 30}
    window.start_collect()
    window.command_timer.stop()
    assert window._recorder is None
    assert window.btn_record.isEnabled()
    window.toggle_recording()
    return window._recorder


def feed(window, count=4):
    window.serial_port.payload = b"".join(
        bp_pack(BP_TYPE_EEG, index, b"\0" * 3 + struct.pack("<8i", *([index + 1] * 8)))
        for index in range(count)
    )
    window.update_show()


@pytest.mark.parametrize("ending", ["stop", "disconnect", "closed_port", "close", "abort"])
def test_window_lifecycle_closes_tail_once(window, ending):
    recorder = start(window)
    feed(window)
    if ending == "stop":
        window.stop_collection()
    elif ending == "disconnect":
        window._handle_serial_disconnect()
    elif ending == "closed_port":
        window.serial_port.is_open = False
        window.update_show()
    elif ending == "abort":
        window._abort_stream_start()
    else:
        window.close()
    window.stop_collection()
    assert recorder._stream is None
    assert not recorder._thread.is_alive()
    assert len(window.recorded_files) == 1
    loaded = read_history_csv(window.recorded_files[0], CHANNELS)
    np.testing.assert_allclose(loaded[0], np.round(np.arange(1, 5) * ADC_COUNT_TO_UV, 3))
    assert "共 1 个文件" in window.record_status_label.text()
    assert window.recorded_files[0] in window.record_status_label.toolTip()


def test_recording_oserror_does_not_stop_acquisition(window, monkeypatch, qt_app):
    recorder = start(window)

    def fail(*args, **kwargs):
        raise OSError("磁盘已满")

    monkeypatch.setattr(recorder, "_open_segment", fail)
    feed(window, 256)
    assert recorder._failed.wait(3)
    qt_app.processEvents()
    assert window.is_collecting
    assert "失败" in window.record_status_label.text()
    feed(window, 2)
    assert window.is_collecting
    assert len(window.AdsData.emg_data[0]) == 258
    window.stop_collection()
    assert recorder.error == "磁盘已满"
    assert not recorder._thread.is_alive()


def test_filter_switch_only_records_new_display_samples(window):
    recorder = start(window)
    feed(window, 4)
    window.btn_filter.setChecked(True)
    window.toggle_filter(True)
    feed(window, 3)
    expected = [channel[4:] for channel in window.filtered_emg_data]
    window.stop_collection()
    assert len(recorder.paths) == 2
    assert "raw_001" in recorder.paths[0]
    assert "filtered_002" in recorder.paths[1]
    assert len(read_history_csv(recorder.paths[0], CHANNELS)[0]) == 4
    np.testing.assert_allclose(read_history_csv(recorder.paths[1], CHANNELS),
                               np.round(expected, 3), atol=1e-10)


def test_directory_settings_persist_and_history_does_not_record(window, tmp_path, monkeypatch):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QFileDialog

    chosen = str(tmp_path / "chosen")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: chosen)
    window.select_record_directory()
    window.record_seconds.setValue(60)
    window.record_max_rows.setValue(100)
    settings = QSettings("EEG_PET_DESK_ZC", "Recording")
    assert settings.value("directory") == chosen
    assert settings.value("segment_seconds", type=int) == 60
    assert settings.value("max_rows", type=int) == 100
    window.is_history_mode = True
    window._start_auto_recording()
    assert window._recorder is None


def test_subject_filename_for_person_and_pet(tmp_path):
    person = {"kind": "person", "name": "张三", "age": 30}
    pet = {"kind": "pet", "name": "旺财", "age": 4, "weight": 12.5}
    assert subject_filename_prefix(person, START) == "张三-30-20261029_123456"
    assert subject_filename_prefix(pet, START) == "旺财-4-12.5kg-20261029_123456"
    assert subject_filename_prefix({**person, "name": "A/B"}, START).startswith("A_B-")
    assert subject_filename_prefix({**person, "name": "CON"}, START).startswith("_CON-")
    for subject, prefix in ((person, "张三-30"), (pet, "旺财-4-12.5kg")):
        recorder = SegmentedCsvRecorder(tmp_path, max_rows=2, started_at=START,
                                       subject=subject)
        recorder.submit(rows(3), 250, "raw", CHANNELS)
        paths = recorder.close()
        assert [Path(path).name for path in paths] == [
            f"{prefix}-20261029_123456_250sps_raw_001.csv",
            f"{prefix}-20261029_123456_250sps_raw_002.csv",
        ]
        assert sum(len(read_history_csv(path, CHANNELS)[0]) for path in paths) == 3


def test_subject_dialog_fields_and_validation(window, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    editor = window._test_subject_dialog_class(window)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    editor._accept_validated()
    assert warnings == ["请选择人的采集或宠物的采集"]
    editor.kind.setCurrentIndex(2)
    assert not editor.weight.isHidden()
    editor._accept_validated()
    assert warnings[-1] == "请输入姓名或名称"
    editor.name.setText("旺财")
    editor._accept_validated()
    assert warnings[-1] == "请输入年龄"
    editor.age.setValue(4)
    editor._accept_validated()
    assert warnings[-1] == "请输入宠物体重"
    editor.weight.setValue(12.5)
    editor._accept_validated()
    assert editor.subject() == {"kind": "pet", "name": "旺财", "age": 4, "weight": 12.5}


def test_collection_requires_subject_and_recording_is_opt_in(window, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QMessageBox
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.serial_port = SerialStub()
    window.start_collect()
    assert warnings == ["未录入被采集者信息"]
    assert not window.is_collecting
    window.subject_info = {"kind": "person", "name": "李四", "age": 28}
    window.start_collect()
    window.command_timer.stop()
    assert window.is_collecting
    feed(window, 3)
    assert window._recorder is None
    assert not list(tmp_path.rglob("*.csv"))
    window.toggle_recording()
    feed(window, 4)
    window.toggle_recording()
    assert window.is_collecting
    assert len(window.recorded_files) == 1
    assert len(read_history_csv(window.recorded_files[0], CHANNELS)[0]) == 4
    assert Path(window.recorded_files[0]).name.startswith("李四-28-")
    assert window.btn_record.text() == "开始保存数据"
    window.stop_collection()
    assert window.subject_info is None
    window.start_collect()
    assert warnings[-1] == "未录入被采集者信息"


def test_recording_can_restart_without_backfilling(window):
    window.serial_port = SerialStub()
    window.subject_info = {"kind": "pet", "name": "旺财", "age": 4, "weight": 12.5}
    window.start_collect()
    window.command_timer.stop()
    feed(window, 2)
    window.toggle_recording()
    feed(window, 3)
    window.toggle_recording()
    feed(window, 2)
    window.toggle_recording()
    feed(window, 4)
    window.stop_collection()
    assert len(window.recorded_files) == 2
    assert [len(read_history_csv(path, CHANNELS)[0]) for path in window.recorded_files] == [3, 4]
    assert all(Path(path).name.startswith("旺财-4-12.5kg-") for path in window.recorded_files)
    assert len(set(window.recorded_files)) == 2


def test_manual_live_export_suggests_subject_filename(window, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    window.serial_port = SerialStub()
    window.subject_info = {"kind": "person", "name": "李四", "age": 28}
    window.start_collect()
    window.command_timer.stop()
    feed(window, 1)
    defaults = []

    def inspect(*args):
        defaults.append(args[2])
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", inspect)
    window.export_csv()
    assert defaults[0].startswith("李四-28-")
    assert defaults[0].endswith("_250sps_raw.csv")


def test_export_preserves_format_and_snapshot_lengths(tmp_path, qt_app):
    path = tmp_path / "export.csv"
    channels = [[1.23456, None, "invalid"], [2]] + [[] for _ in range(6)]
    worker = CsvExportWorker(path, channels, CHANNELS)
    channels[0].append(999)
    worker.start()
    assert worker.wait(5000)
    assert worker.error is None
    assert not worker.cancelled
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.reader(stream))
    assert actual == [CHANNELS, ["1.235", "2.000"] + [""] * 6, [""] * 8, [""] * 8]


def test_export_cancel_removes_partial_file(tmp_path, qt_app, monkeypatch):
    from csv_recording import format_csv_row
    import csv_recording

    path = tmp_path / "cancelled.csv"
    worker = CsvExportWorker(path, [[1.0] * 10000 for _ in range(8)], CHANNELS)
    entered, release = threading.Event(), threading.Event()

    def pause(values):
        entered.set()
        release.wait(3)
        return format_csv_row(values)

    monkeypatch.setattr(csv_recording, "format_csv_row", pause)
    worker.start()
    assert entered.wait(3)
    worker.requestInterruption()
    release.set()
    assert worker.wait(5000)
    assert worker.cancelled
    assert not path.exists()


def test_export_dialog_disables_button_and_cancel_cleans_file(window, tmp_path, monkeypatch, qt_app):
    import csv_recording
    from PyQt5.QtWidgets import QFileDialog

    path = tmp_path / "dialog_cancel.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    for channel in window.AdsData.emg_data:
        channel.extend([1.0] * 1000)
    entered, release = threading.Event(), threading.Event()
    original_format = csv_recording.format_csv_row

    def pause(values):
        entered.set()
        release.wait(3)
        return original_format(values)

    monkeypatch.setattr(csv_recording, "format_csv_row", pause)
    window.export_csv()
    assert not window.btn_export.isEnabled()
    assert window._export_progress.isVisible()
    assert entered.wait(3)
    window._export_progress.canceled.emit()
    release.set()
    deadline = time.monotonic() + 5
    while window._export_worker is not None and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    assert window._export_worker is None
    assert window.btn_export.isEnabled()
    assert not path.exists()


def test_spec_icon_is_independent_of_working_directory(tmp_path, monkeypatch):
    path = ROOT / "python_ads1299/main_uart_v1.2.spec"
    source = path.read_text(encoding="utf-8").split("a = Analysis(")[0]
    monkeypatch.chdir(tmp_path)
    scope = {"SPECPATH": str(path.parent)}
    exec(compile(source, str(path), "exec"), scope)
    assert Path(scope["ICON_PATH"]) == path.parent / "mouse.ico"
    scope = {"SPECPATH": str(tmp_path)}
    exec(compile(source, str(path), "exec"), scope)
    assert scope["ICON_PATH"] is None
