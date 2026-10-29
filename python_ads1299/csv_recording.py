"""后台增量记录，避免磁盘延迟阻塞串口读取和界面刷新。"""

import csv
import datetime
import re
import queue
import threading
import time
from numbers import Real
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal


def format_csv_row(values):
    return [f"{float(value):.3f}" if isinstance(value, Real) else ""
            for value in values]


def safe_filename_part(value):
    # 用户输入只用于文件名，不能形成路径或 Windows 保留文件名。
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value).strip())
    cleaned = cleaned.rstrip(" .")[:80] or "未命名"
    if cleaned.split(".", 1)[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        cleaned = "_" + cleaned
    return cleaned


def subject_filename_prefix(subject, started_at):
    name = safe_filename_part(subject["name"])
    age = int(subject["age"])
    fields = [name, str(age)]
    if subject["kind"] == "pet":
        weight = f"{float(subject['weight']):.2f}".rstrip("0").rstrip(".")
        fields.append(f"{weight}kg")
    fields.append(started_at.strftime("%Y%m%d_%H%M%S"))
    return "-".join(fields)


def segment_filename(started_at, sample_rate, source, sequence, subject=None):
    prefix = (subject_filename_prefix(subject, started_at) if subject else
              started_at.strftime("%Y%m%d_%H%M%S"))
    return (f"{prefix}_{int(sample_rate)}sps_"
            f"{source}_{sequence:03d}.csv")


class SegmentedCsvRecorder:
    """每个会话拥有独立的队列和文件句柄，停止时排空尾批再关闭。"""

    def __init__(self, directory, segment_seconds=300, max_rows=0,
                 on_path=None, on_error=None, started_at=None, subject=None):
        self.started_at = started_at or datetime.datetime.now()
        self.subject = dict(subject) if subject else None
        self.directory = Path(directory) / self.started_at.strftime("%Y%m%d")
        self.segment_seconds = max(1, int(segment_seconds))
        self.max_rows = max(0, int(max_rows))
        self.on_path = on_path or (lambda path: None)
        self.on_error = on_error or (lambda message: None)
        self.paths = []
        self.error = None
        self._queue = queue.Queue(maxsize=128)
        self._failed = threading.Event()
        self._closing = False
        self._stream = None
        self._writer = None
        self._metadata = None
        self._sequence = 0
        self._rows = 0
        self._pending = []
        self._last_flush = time.monotonic()
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, rows, sample_rate, source, headers):
        if self._closing or self._failed.is_set() or not rows:
            return False
        try:
            self._queue.put_nowait((rows, (int(sample_rate), source, tuple(headers))))
        except queue.Full:
            # 慢盘不能反向阻塞采集，也不能静默丢失一部分记录。
            self._fail(OSError("写盘队列已满，本次自动记录已停止"))
            return False
        return True

    def close(self):
        if not self._closing:
            self._closing = True
            self._queue.put(None)
        self._thread.join()
        return list(self.paths)

    def _fail(self, exc):
        with self._error_lock:
            if self._failed.is_set():
                return
            self.error = str(exc)
            self._failed.set()
        self.on_error(self.error)

    def _close_stream(self):
        stream, self._stream = self._stream, None
        self._writer = None
        if stream is not None:
            stream.close()

    def _open_segment(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        rate, source, headers = self._metadata
        while True:
            self._sequence += 1
            path = self.directory / segment_filename(
                self.started_at, rate, source, self._sequence,
                self.subject,
            )
            try:
                # 同秒重启会话或目录内已有文件时，也不能覆盖之前的记录。
                self._stream = path.open("x", newline="", encoding="utf-8-sig")
                break
            except FileExistsError:
                continue
        self._writer = csv.writer(self._stream)
        self._writer.writerow(headers)
        self._rows = 0
        self.paths.append(str(path.resolve()))
        self.on_path(self.paths[-1])

    def _flush_pending(self):
        if not self._pending:
            return
        rate = self._metadata[0]
        limit = max(1, int(rate * self.segment_seconds))
        if self.max_rows:
            limit = min(limit, self.max_rows)
        offset = 0
        while offset < len(self._pending):
            if self._stream is None:
                self._open_segment()
            count = min(limit - self._rows, len(self._pending) - offset)
            self._writer.writerows(format_csv_row(row)
                                   for row in self._pending[offset:offset + count])
            self._stream.flush()
            self._rows += count
            offset += count
            if self._rows == limit:
                self._close_stream()
        self._pending.clear()
        self._last_flush = time.monotonic()

    def _run(self):
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    item = ()
                if item is None:
                    if not self._failed.is_set():
                        self._flush_pending()
                    break
                if self._failed.is_set():
                    self._pending.clear()
                    self._close_stream()
                    continue
                if item:
                    rows, metadata = item
                    if metadata != self._metadata:
                        self._flush_pending()
                        self._close_stream()
                        self._metadata = metadata
                    self._pending.extend(rows)
                batch_size = min(256, max(1, self._metadata[0])) if self._metadata else 256
                if (len(self._pending) >= batch_size
                        or time.monotonic() - self._last_flush >= 1.0):
                    self._flush_pending()
        except OSError as exc:
            self._fail(exc)
            try:
                self._close_stream()
            except OSError:
                pass
            # 写盘失败后继续消费队列，确保停止操作不会因队列已满而阻塞。
            while self._queue.get() is not None:
                pass
        finally:
            try:
                self._close_stream()
            except OSError as exc:
                self._fail(exc)


class CsvExportWorker(QThread):
    progress = pyqtSignal(int)

    def __init__(self, path, channels, headers, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        # 仅固定引用与长度；采集中追加数据或切换缓存不应改变本次导出范围。
        self.channels = tuple(channels)
        self.lengths = tuple(len(channel) for channel in channels)
        self.headers = tuple(headers)
        self.cancelled = False
        self.error = None

    def run(self):
        created = False
        try:
            with self.path.open("w", newline="", encoding="utf-8-sig") as stream:
                created = True
                writer = csv.writer(stream)
                writer.writerow(self.headers)
                total = max(self.lengths, default=0)
                for start in range(0, total, 256):
                    if self.isInterruptionRequested():
                        self.cancelled = True
                        break
                    end = min(total, start + 256)
                    writer.writerows(
                        format_csv_row(channel[index] if index < length else None
                                       for channel, length in zip(self.channels, self.lengths))
                        for index in range(start, end)
                    )
                    self.progress.emit(int(end * 100 / total))
                if self.isInterruptionRequested():
                    self.cancelled = True
        except OSError as exc:
            self.error = str(exc)
        finally:
            if created and (self.cancelled or self.error):
                try:
                    self.path.unlink()
                except OSError as exc:
                    self.error = f"{self.error or '导出已取消'}；无法删除未完成文件：{exc}"
