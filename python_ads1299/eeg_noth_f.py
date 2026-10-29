"""EEG 二阶 IIR 陷波滤波器设计，兼容原 MATLAB 参数含义。"""

from __future__ import annotations

import numpy as np


def eeg_noth_f(
    fs: float,
    notch_f: float,
    bandwidth: float = 5.0,
    attenuation_db: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成二阶 IIR 陷波滤波器系数。

    该实现对应原 MATLAB 调用::

        iirnotch(notch_f/(fs/2), bandwidth/(fs/2), attenuation_db)

    原文件中的默认值为 5 Hz 带宽，并在 -2 dB 电平处定义带宽。
    SciPy 的 ``iirnotch`` 只直接接收 -3 dB 带宽，因此这里显式使用
    MATLAB/Orfanidis 形式计算，以保留第三个参数的含义。

    参数
    ----
    fs:
        采样率，单位 Hz。
    notch_f:
        陷波中心频率，单位 Hz。
    bandwidth:
        在 ``-attenuation_db`` dB 电平处定义的带宽，单位 Hz。
    attenuation_db:
        带宽对应的衰减电平绝对值，单位 dB；原 MATLAB 文件为 2 dB。

    返回
    ----
    b, a:
        IIR 滤波器分子、分母系数，可传给 ``scipy.signal.lfilter``。
    """
    fs = float(fs)
    notch_f = float(notch_f)
    bandwidth = float(bandwidth)
    attenuation_db = float(attenuation_db)

    if fs <= 0:
        raise ValueError("fs 必须大于 0 Hz")
    if not 0 < notch_f < fs / 2:
        raise ValueError(f"notch_f 必须位于 (0, {fs / 2}) Hz 内")
    if not 0 < bandwidth < fs / 2:
        raise ValueError(f"bandwidth 必须位于 (0, {fs / 2}) Hz 内")
    if attenuation_db <= 0:
        raise ValueError("attenuation_db 必须大于 0 dB")

    # 归一化到 Nyquist = 1，与 MATLAB iirnotch 的 w0、bw 定义一致。
    w0 = notch_f / (fs / 2.0)
    bw = bandwidth / (fs / 2.0)

    # 在 -attenuation_db dB 处的线性幅值。
    gb = 10.0 ** (-attenuation_db / 20.0)
    beta = (np.sqrt(1.0 - gb * gb) / gb) * np.tan(np.pi * bw / 2.0)
    gain = 1.0 / (1.0 + beta)
    cos_w0 = np.cos(np.pi * w0)

    b = gain * np.array([1.0, -2.0 * cos_w0, 1.0], dtype=np.float64)
    a = np.array(
        [1.0, -2.0 * gain * cos_w0, 2.0 * gain - 1.0],
        dtype=np.float64,
    )
    return b, a
