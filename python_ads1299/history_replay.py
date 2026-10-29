"""CSV replay loading and narrow-band EEG activity detection."""

import csv
import os
import re

import numpy as np


def sample_rate_from_filename(path):
    match = re.search(r"_(\d+)sps(?:_|\.)", os.path.basename(path), re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_history_csv(path, channel_names, header_aliases=()):
    """Read the eight-channel CSV format written by the main window."""
    channels = [[] for _ in channel_names]
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        headers = next(reader, None)
        if headers is None:
            raise ValueError("CSV 文件为空")
        headers = [header.strip() for header in headers]
        candidates = (channel_names, *header_aliases)
        matching = next((names for names in candidates
                         if len(headers) == len(names) and set(headers) == set(names)), None)
        if matching is None:
            raise ValueError("CSV 表头必须包含八个通道：" + ", ".join(channel_names))
        column_indices = [headers.index(name) for name in matching]
        for row_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != len(headers):
                raise ValueError(f"第 {row_number} 行的通道数不正确")
            try:
                values = [float(row[index]) for index in column_indices]
            except ValueError as exc:
                raise ValueError(f"第 {row_number} 行包含空值或非数字") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(f"第 {row_number} 行包含无效数值")
            for channel, value in zip(channels, values):
                channel.append(value)
    if not channels[0]:
        raise ValueError("CSV 文件中没有脑电采样数据")
    return [np.asarray(channel, dtype=np.float64) for channel in channels]


def alpha_activity_mask(values, sample_rate, sample_offset=0, band=(9.0, 13.0)):
    """Mark 2 s windows dominated by the selected energy band in 1-40 Hz."""
    low, high = map(float, band)
    if not 1.0 <= low < high <= 40.0 or high - low < 0.5:
        raise ValueError("检测频段须在 1–40 Hz 内，且至少宽 0.5 Hz")
    signal = np.asarray(values, dtype=np.float64)
    mask = np.zeros(signal.size, dtype=bool)
    if sample_rate < 80 or signal.size < int(round(2 * sample_rate)):
        return mask

    window_size = int(round(2 * sample_rate))
    hop_size = max(1, int(round(0.5 * sample_rate)))
    window = np.hanning(window_size)
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    reference = (frequencies >= 1.0) & (frequencies <= 40.0)
    target = (frequencies >= low) & (frequencies <= high)

    first_global_start = ((sample_offset + hop_size - 1) // hop_size) * hop_size
    first_start = first_global_start - sample_offset
    for start in range(first_start, signal.size - window_size + 1, hop_size):
        segment = signal[start:start + window_size]
        if not np.all(np.isfinite(segment)):
            continue
        centered = segment - np.mean(segment)
        if np.std(centered) < 0.5:
            continue
        power = np.abs(np.fft.rfft(centered * window)) ** 2
        total_power = np.sum(power[reference])
        if total_power <= 0:
            continue
        peak_frequency = frequencies[reference][np.argmax(power[reference])]
        if low <= peak_frequency <= high and np.sum(power[target]) / total_power >= 0.45:
            mask[start:start + window_size] = True
    return mask
