"""EEG 4 阶 Butterworth 高通滤波器设计。"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter


def eeg_hp_f(fs: float, fc: float) -> tuple[np.ndarray, np.ndarray]:
    """生成 4 阶 Butterworth 高通滤波器系数。

    参数
    ----
    fs:
        采样率，单位 Hz。
    fc:
        -3 dB 截止频率，单位 Hz。

    返回
    ----
    b, a:
        IIR 滤波器分子、分母系数，可传给 ``scipy.signal.lfilter``。
    """
    fs = float(fs)
    fc = float(fc)

    if fs <= 0:
        raise ValueError("fs 必须大于 0 Hz")
    if not 0 < fc < fs / 2:
        raise ValueError(f"fc 必须位于 (0, {fs / 2}) Hz 内")

    b, a = butter(N=4, Wn=fc, btype="highpass", fs=fs, output="ba")
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)
