"""Offline alpha and mu rhythm measurements for eight-channel EEG recordings."""

from __future__ import annotations

import csv
import re

import numpy as np
from scipy import signal


BAND = (8.0, 13.0)
CENTRAL_PREFIX = ("C", "FC", "CP")
OCCIPITAL_PREFIX = ("O", "PO", "P")
DROP_COLUMNS = {"time", "timestamp", "t", "index", "marker", "event"}


def read_analysis_csv(path, known_names=(), aliases=()):
    """Load named EEG columns, preserving current channel identities when known."""
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        headers = next(reader, None)
        if not headers:
            raise ValueError("CSV 文件为空")
        headers = [header.strip() for header in headers]
        if len(headers) != len(set(headers)) or any(not header for header in headers):
            raise ValueError("CSV 表头包含重复或空通道名")
        indices = [index for index, header in enumerate(headers)
                   if header.lower() not in DROP_COLUMNS]
        if not indices:
            raise ValueError("CSV 中没有 EEG 通道")
        source_names = [headers[index] for index in indices]
        matching = next((names for names in (known_names, *aliases)
                         if len(source_names) == len(names)
                         and set(source_names) == set(names)), None)
        if matching is not None:
            indices = [headers.index(name) for name in matching]
            names = list(known_names)
        else:
            names = source_names
        rows = []
        for number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV 第 {number} 行列数不正确")
            try:
                rows.append([float(row[index]) for index in indices])
            except ValueError as exc:
                raise ValueError(f"CSV 第 {number} 行包含非数字或空值") from exc
    if not rows:
        raise ValueError("CSV 中没有 EEG 采样数据")
    data = np.asarray(rows, dtype=np.float64).T
    if not np.all(np.isfinite(data)):
        raise ValueError("CSV 包含无效数值")
    return names, data, matching is not None


def parse_ranges(text, duration):
    """Parse comma-separated second ranges, merging overlaps."""
    if not text.strip():
        return []
    ranges = []
    for part in text.split(","):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", part)
        if match is None:
            raise ValueError(f"时间段格式不正确：{part.strip() or '(空)'}")
        start, end = map(float, match.groups())
        if start >= end or end > duration + 1e-6:
            raise ValueError(f"时间段 {part.strip()} 超出 0–{duration:.1f} 秒或起止顺序错误")
        ranges.append((start, min(end, duration)))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _welch_runs(channels, fs, runs):
    """Average PSDs without joining data across rejected intervals."""
    total = None
    weight = 0
    for start, end in runs:
        block = channels[:, start:end]
        if block.shape[1] < 2 * fs:
            continue
        block = block - np.mean(block, axis=1, keepdims=True)
        frequencies, psd = signal.welch(
            block, fs=fs, axis=1, nperseg=min(block.shape[1], 4 * fs),
            noverlap=None, nfft=4 * fs,
        )
        amount = block.shape[1]
        total = psd * amount if total is None else total + psd * amount
        weight += amount
    return (frequencies, total / weight) if weight else (None, None)


def _power(frequencies, psd, low, high):
    mask = (frequencies >= low) & (frequencies <= high)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def _peak(frequencies, psd, low, high):
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    index = indices[int(np.argmax(psd[mask]))]
    if psd[index] <= 0:
        return None
    return float(frequencies[index]), float(psd[index]), int(index)


def _alpha_peak(frequencies, psd):
    peak = _peak(frequencies, psd, 7.0, 14.0)
    if peak is None:
        return None
    frequency, power, index = peak
    background = psd[(frequencies >= 6.0) & (frequencies <= 20.0)]
    median = float(np.median(background)) if background.size else 0.0
    left = right = index
    while left > 0 and psd[left] > power / 2:
        left -= 1
    while right < len(psd) - 1 and psd[right] > power / 2:
        right += 1
    bandwidth = float(frequencies[right] - frequencies[left])
    return frequency, power / median if median > 0 else 0.0, bandwidth


def _arcade_index(values, fs):
    """Estimate alternating half-cycle asymmetry in a broad 3-30 Hz signal."""
    if len(values) < 2 * fs:
        return 0.0
    high = min(30.0, fs / 2 - 1.0)
    filtered = signal.sosfiltfilt(
        signal.butter(4, [3.0, high], btype="bandpass", fs=fs, output="sos"),
        values - np.mean(values),
    )
    signs = np.flatnonzero(filtered[:-1] * filtered[1:] < 0)
    if len(signs) < 9:
        return 0.0
    crossings = signs + np.abs(filtered[signs]) / (
        np.abs(filtered[signs]) + np.abs(filtered[signs + 1])
    )
    intervals = np.diff(crossings) / fs
    intervals = intervals[(intervals > 1 / 32) & (intervals < 1 / 10)]
    if len(intervals) < 8:
        return 0.0
    count = len(intervals) // 2
    odd = float(np.median(intervals[:2 * count:2]))
    even = float(np.median(intervals[1:2 * count:2]))
    return abs(odd - even) / (odd + even) if odd + even else 0.0


def _spindle_metrics(values, fs, peak_frequency, runs):
    low = max(7.0, peak_frequency - 2.0)
    high = min(14.0, peak_frequency + 2.0)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    events = []
    for start, end in runs:
        if end - start < 2 * fs:
            continue
        filtered = signal.sosfiltfilt(sos, values[start:end] - np.mean(values[start:end]))
        envelope = np.abs(signal.hilbert(filtered))
        width = max(1, round(0.08 * fs))
        envelope = np.convolve(envelope, np.ones(width) / width, mode="same")
        baseline = float(np.median(envelope))
        mad = float(np.median(np.abs(envelope - baseline))) * 1.4826
        above = envelope > max(baseline + 2.0 * mad, baseline * 1.8)
        edges = np.diff(np.r_[False, above, False].astype(int))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if 0.3 <= (right - left) / fs <= 3.0:
                events.append(((right - left) / fs, float(np.ptp(filtered[left:right]))))
    if not events:
        return 0, None, None
    return len(events), float(np.mean([event[0] for event in events])), float(
        np.mean([event[1] for event in events]),
    )


def _condition_runs(ranges, good_runs, fs):
    selected = []
    for first, last in ranges:
        left, right = round(first * fs), round(last * fs)
        for start, end in good_runs:
            a, b = max(left, start), min(right, end)
            if b - a >= 2 * fs:
                selected.append((a, b))
    return selected


def _condition_psd(channels, fs, ranges, good_runs):
    return _welch_runs(channels, fs, _condition_runs(ranges, good_runs, fs))[1]


def _erd_scan(values, fs, runs):
    best = None
    for start, end in runs:
        if end - start < 8 * fs:
            continue
        powers = []
        times = []
        for position in range(start, end - 3 * fs + 1, fs):
            frequencies, psd = signal.welch(
                values[position:position + 3 * fs], fs=fs,
                nperseg=3 * fs, nfft=4 * fs,
            )
            powers.append(_power(frequencies, psd, *BAND))
            times.append((position + 1.5 * fs) / fs)
        if len(powers) < 5:
            continue
        median = float(np.median(powers))
        if median <= 0:
            continue
        index = int(np.argmin(powers))
        drop = (powers[index] - median) / median * 100
        if best is None or drop < best["change_pct"]:
            best = {"time_s": times[index], "change_pct": drop}
    return best


def analyze_rhythms(data, sample_rate, channel_names, bad_threshold=200.0,
                    mode="both", segments=None, scan=True, hand="right"):
    """Return spectrum, alpha and mu metrics for a stable data snapshot."""
    fs = int(round(sample_rate))
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(channel_names):
        raise ValueError("数据须按通道排列，且与通道名称数量一致")
    if not 80 <= fs <= 4000 or bad_threshold <= 0:
        raise ValueError("采样率或伪迹阈值无效")
    if mode not in ("alpha", "mu", "both") or hand not in ("left", "right", "both"):
        raise ValueError("分析模式无效")
    if values.shape[1] < 4 * fs:
        raise ValueError("至少需要 4 秒脑电数据")
    if not np.all(np.isfinite(values)):
        raise ValueError("数据包含空值或无效数值")
    duration = values.shape[1] / fs
    segments = {key: parse_ranges((segments or {}).get(key, ""), duration)
                for key in ("closed", "open", "rest", "move", "post", "eyesopen")}

    whole_seconds = values.shape[1] // fs
    windows = values[:, :whole_seconds * fs].reshape(len(channel_names), whole_seconds, fs)
    centered = windows - np.median(windows, axis=2, keepdims=True)
    maxima = np.max(np.abs(centered), axis=(0, 2))
    bad = maxima > bad_threshold
    edges = np.diff(np.r_[False, ~bad, False].astype(int))
    good_runs = [(int(a * fs), int(b * fs)) for a, b in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1),
    ) if b - a >= 2]
    good_seconds = sum((end - start) / fs for start, end in good_runs)
    if good_seconds < 4:
        raise ValueError("剔除伪迹后连续可用数据不足 4 秒")

    frequencies, psds = _welch_runs(values, fs, good_runs)
    names = list(channel_names)
    central = [name for name in names if name.upper().startswith(CENTRAL_PREFIX)]
    occipital = [name for name in names if name.upper().startswith(OCCIPITAL_PREFIX)]
    notes = []
    if mode in ("alpha", "both") and not occipital:
        notes.append("无枕顶区导联；额区数据不能单独确认典型 α 节律。")
    if mode in ("mu", "both") and not central:
        notes.append("无中央区导联；μ 相关结果仅供参考，建议采集 C3/Cz/C4。")
    if central and hand != "both":
        side = ("C3", "C1", "FC3", "CP3", "C5") if hand == "right" else (
            "C4", "C2", "FC4", "CP4", "C6",
        )
        matches = [name for name in central if name.upper() in side]
        if matches:
            notes.append(f"{('右' if hand == 'right' else '左')}手运动的对侧导联：{', '.join(matches)}")

    spectrum = []
    for index, name in enumerate(names):
        psd = psds[index]
        peak = _peak(frequencies, psd, *BAND)
        background = psd[(frequencies >= 6) & (frequencies <= 30)]
        median = float(np.median(background)) if background.size else 0.0
        ratio = 100 * _power(frequencies, psd, *BAND) / max(
            _power(frequencies, psd, 1, 45), 1e-12,
        )
        harmonic = None
        if peak is not None and 14 <= 2 * peak[0] <= 30:
            candidate = _peak(frequencies, psd, 2 * peak[0] - 2, 2 * peak[0] + 2)
            if candidate and candidate[1] > 1.5 * median:
                harmonic = candidate[0]
        asymmetry = np.average([
            _arcade_index(values[index, start:end], fs) for start, end in good_runs
        ], weights=[end - start for start, end in good_runs])
        spectrum.append({"channel": name, "peak_hz": peak[0] if peak else None,
                         "saliency": peak[1] / median if peak and median > 0 else 0.0,
                         "harmonic_hz": harmonic, "band_ratio_pct": ratio,
                         "arcade_index": float(asymmetry)})

    alpha = []
    berger = None
    if mode in ("alpha", "both"):
        if segments["closed"] and segments["open"]:
            closed = _condition_psd(values, fs, segments["closed"], good_runs)
            opened = _condition_psd(values, fs, segments["open"], good_runs)
            if closed is None or opened is None:
                notes.append("睁眼或闭眼可用段不足 2 秒，已跳过 Berger 阻断分析。")
            else:
                berger = (closed, opened)
        for index, row in enumerate(spectrum):
            peak = _alpha_peak(frequencies, psds[index])
            iaf, saliency, bandwidth = peak if peak else (None, 0.0, None)
            detected = iaf is not None and BAND[0] <= iaf <= BAND[1]
            verdict = ("检出 α" if detected and saliency >= 2 and row["band_ratio_pct"] >= 8
                       else "可疑" if detected else "未检出")
            spindle_count, spindle_duration, spindle_amplitude = _spindle_metrics(
                values[index], fs, iaf or 10.0, good_runs,
            )
            suppression = None
            if berger:
                baseline = _power(frequencies, berger[0][index], *BAND)
                opened_power = _power(frequencies, berger[1][index], *BAND)
                suppression = 100 * (baseline - opened_power) / baseline if baseline > 0 else None
            berger_verdict = ("典型阻断" if suppression is not None and suppression >= 50 else
                              "部分阻断" if suppression is not None and suppression >= 20 else
                              "无阻断" if suppression is not None else None)
            alpha.append({"channel": row["channel"], "iaf_hz": iaf,
                          "saliency": saliency, "bandwidth_hz": bandwidth,
                          "band_ratio_pct": row["band_ratio_pct"],
                          "spindles": spindle_count,
                          "spindles_per_min": spindle_count * 60 / good_seconds,
                          "spindle_duration_s": spindle_duration,
                          "spindle_amp_uv": spindle_amplitude,
                          "berger_pct": suppression,
                          "berger_verdict": berger_verdict,
                          "verdict": verdict})

    mu = []
    scans = []
    if mode in ("mu", "both"):
        rest = move = post = eyes = None
        if segments["rest"] and segments["move"]:
            rest = _condition_psd(values, fs, segments["rest"], good_runs)
            move = _condition_psd(values, fs, segments["move"], good_runs)
            if rest is None or move is None:
                notes.append("静息或运动可用段不足 2 秒，已跳过 ERD 分析。")
                rest = move = None
            elif segments["post"]:
                post = _condition_psd(values, fs, segments["post"], good_runs)
                if post is None:
                    notes.append("运动后可用段不足 2 秒，已跳过 β 回弹分析。")
            if rest is not None and segments["eyesopen"]:
                eyes = _condition_psd(values, fs, segments["eyesopen"], good_runs)
                if eyes is None:
                    notes.append("睁眼可用段不足 2 秒，已跳过 μ/α 判别。")
        if rest is not None and move is not None:
            for index, row in enumerate(spectrum):
                baseline = _power(frequencies, rest[index], *BAND)
                moving = _power(frequencies, move[index], *BAND)
                erd = 100 * (moving - baseline) / baseline if baseline > 0 else None
                beta_rest = _power(frequencies, rest[index], 20, 30)
                beta_post = _power(frequencies, post[index], 20, 30) if post is not None else None
                ers = (100 * (beta_post - beta_rest) / beta_rest
                       if beta_post is not None and beta_rest > 0 else None)
                eyes_power = _power(frequencies, eyes[index], *BAND) if eyes is not None else None
                eye_suppression = (100 * (baseline - eyes_power) / baseline
                                   if eyes_power is not None and baseline > 0 else None)
                verdict = ("显著 ERD" if erd is not None and erd <= -40 else
                           "中等 ERD" if erd is not None and erd <= -20 else
                           "弱 ERD" if erd is not None and erd < 0 else "无 ERD")
                distinction = None
                if eye_suppression is not None and erd is not None:
                    distinction = ("偏 μ" if erd <= -30 and eye_suppression < 30 else
                                   "偏 α" if eye_suppression >= 50 else "不明确")
                mu.append({"channel": row["channel"], "rest_power": baseline,
                           "move_power": moving, "erd_pct": erd,
                           "beta_ers_pct": ers, "eye_suppression_pct": eye_suppression,
                           "arcade_index": row["arcade_index"],
                           "verdict": verdict, "distinction": distinction})
        elif scan:
            target_names = central or names
            for name in target_names:
                candidate = _erd_scan(values[names.index(name)], fs, good_runs)
                if candidate:
                    scans.append({"channel": name, **candidate})
            notes.append("自动扫描只给出 ERD 候选；确认运动想象反应需录入静息与运动时间段。")
        else:
            notes.append("未提供静息和运动时间段，已跳过 μ ERD 分析。")

    return {"sample_rate": fs, "duration_s": duration, "channels": names,
            "bad_fraction": float(np.mean(bad)), "good_seconds": good_seconds,
            "max_uv": maxima, "bad_mask": bad, "good_runs": good_runs,
            "frequencies": frequencies, "psds": psds, "spectrum": spectrum,
            "alpha": alpha, "mu": mu, "scans": scans, "notes": notes,
            "segments": segments, "bad_threshold": bad_threshold, "mode": mode}
