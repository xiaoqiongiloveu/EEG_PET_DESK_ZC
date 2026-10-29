"""实时 EEG 级联滤波器。

对应 MATLAB ``eeg_filters`` 类：

1. 0.5 Hz 四阶 Butterworth 高通；
2. 100 Hz 四阶 Butterworth 低通；
3. 50 Hz 陷波；
4. 当采样率为 500/1000/2000 Hz 时，再增加 100 Hz 陷波。

本类保存每一级 IIR 的最终状态，可连续处理串口分块数据，避免每个数据块
都从零状态重新开始。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi

from eeg_hp_f import eeg_hp_f
from eeg_lp_f import eeg_lp_f
from eeg_noth_f import eeg_noth_f

FilterCoefficients = tuple[np.ndarray, np.ndarray]
FilterState = np.ndarray | None


class eeg_filters:
    """与原 MATLAB 类同名的实时 EEG 滤波器类。"""

    SUPPORTED_FS = (250, 500, 1000, 2000)

    def __init__(self) -> None:
        self.hds_250: list[FilterCoefficients] = [
            eeg_hp_f(250, 0.5),
            eeg_lp_f(250, 100),
            eeg_noth_f(250, 50),
        ]
        self.zf_250: list[FilterState] = [None] * len(self.hds_250)

        self.hds_500: list[FilterCoefficients] = [
            eeg_hp_f(500, 0.5),
            eeg_lp_f(500, 100),
            eeg_noth_f(500, 50),
            eeg_noth_f(500, 100),
        ]
        self.zf_500: list[FilterState] = [None] * len(self.hds_500)

        self.hds_1000: list[FilterCoefficients] = [
            eeg_hp_f(1000, 0.5),
            eeg_lp_f(1000, 100),
            eeg_noth_f(1000, 50),
            eeg_noth_f(1000, 100),
        ]
        self.zf_1000: list[FilterState] = [None] * len(self.hds_1000)

        self.hds_2000: list[FilterCoefficients] = [
            eeg_hp_f(2000, 0.5),
            eeg_lp_f(2000, 100),
            eeg_noth_f(2000, 50),
            eeg_noth_f(2000, 100),
        ]
        self.zf_2000: list[FilterState] = [None] * len(self.hds_2000)

        self._filters: dict[int, list[FilterCoefficients]] = {
            250: self.hds_250,
            500: self.hds_500,
            1000: self.hds_1000,
            2000: self.hds_2000,
        }
        self._states: dict[int, list[FilterState]] = {
            250: self.zf_250,
            500: self.zf_500,
            1000: self.zf_1000,
            2000: self.zf_2000,
        }

    @staticmethod
    def _state_shape(data_shape: tuple[int, ...], axis: int, length: int) -> tuple[int, ...]:
        shape = list(data_shape)
        shape[axis] = length
        return tuple(shape)

    def hds_filter(
        self,
        hds: Sequence[FilterCoefficients],
        zfs: list[FilterState],
        data: np.ndarray | Sequence[float],
        axis: int = -1,
    ) -> tuple[np.ndarray, list[FilterState]]:
        """依次应用多个滤波器，并更新每一级的实时状态。

        ``axis`` 指定时间/采样点所在维度：

        - 一维数据 ``(samples,)``：使用默认 ``axis=-1``；
        - 多通道 ``(channels, samples)``：使用默认 ``axis=-1``；
        - 多通道 ``(samples, channels)``：设置 ``axis=0``。
        """
        if len(hds) != len(zfs):
            raise ValueError("hds 与 zfs 的数量必须一致")

        filtered = np.asarray(data, dtype=np.float64)
        if filtered.ndim == 0:
            raise ValueError("data 至少必须是一维数组")

        if axis < 0:
            axis += filtered.ndim
        if not 0 <= axis < filtered.ndim:
            raise np.AxisError(axis, ndim=filtered.ndim)
        if filtered.shape[axis] == 0:
            return filtered.copy(), zfs

        for index, (b, a) in enumerate(hds):
            state_length = max(len(a), len(b)) - 1
            expected_shape = self._state_shape(filtered.shape, axis, state_length)
            zi = zfs[index]

            if zi is None:
                zi = np.zeros(expected_shape, dtype=np.float64)
            elif zi.shape != expected_shape:
                # 通道数或非时间维度发生变化时，旧状态无法复用，安全地重新初始化。
                zi = np.zeros(expected_shape, dtype=np.float64)

            filtered, zf = lfilter(b, a, filtered, axis=axis, zi=zi)
            zfs[index] = zf

        return filtered, zfs

    def data_filter(
        self,
        fs: int,
        data: np.ndarray | Sequence[float],
        axis: int = -1,
    ) -> np.ndarray:
        """按采样率选择滤波器组并处理一块 EEG 数据。"""
        try:
            hds = self._filters[int(fs)]
            zfs = self._states[int(fs)]
        except (KeyError, ValueError, TypeError) as exc:
            supported = ", ".join(str(value) for value in self.SUPPORTED_FS)
            raise ValueError(f"不支持的采样率 {fs!r}，可选值：{supported} Hz") from exc

        filtered, _ = self.hds_filter(hds, zfs, data, axis=axis)
        return filtered

    def reset(self, fs: int | None = None) -> None:
        """清空实时滤波状态。

        ``fs=None`` 清空全部采样率状态；传入具体采样率只清空对应状态。
        更换被试、重新连接设备或开始一段全新记录时建议调用。
        """
        if fs is None:
            sample_rates = self.SUPPORTED_FS
        else:
            try:
                sample_rate = int(fs)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"无效采样率：{fs!r}") from exc
            if sample_rate not in self._states:
                supported = ", ".join(str(value) for value in self.SUPPORTED_FS)
                raise ValueError(f"不支持的采样率 {fs!r}，可选值：{supported} Hz")
            sample_rates = (sample_rate,)

        for sample_rate in sample_rates:
            states = self._states[sample_rate]
            states[:] = [None] * len(states)


# 推荐的 Python 类名，同时保留 MATLAB 风格类名以便直接迁移。
EEGFilters = eeg_filters


class ConfigurableEEGFilter:
    """对八通道连续数据应用可调带通和可选 50 Hz 陷波。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.settings = None
        self.dc_sos = None
        self.bandpass_sos = None
        self.notch_coefficients = None
        self.dc_state = None
        self.bandpass_state = None
        self.notch_state = None

    def configure(self, rate, low, high, bandpass, notch, remove_dc=False):
        rate = int(rate)
        low, high = float(low), float(high)
        if rate <= 0:
            raise ValueError("采样率必须大于 0 Hz")
        if bandpass and not (0 < low < high < rate / 2):
            raise ValueError(f"截止频率必须满足 0 < 下限 < 上限 < {rate / 2:g} Hz")
        if notch and rate <= 100:
            raise ValueError("50 Hz 陷波要求采样率高于 100 Hz")

        settings = (rate, low, high, bool(bandpass), bool(notch), bool(remove_dc))
        if settings == self.settings:
            return
        self.reset()
        self.settings = settings
        if remove_dc and not bandpass:
            self.dc_sos = butter(1, 0.1, btype="highpass", fs=rate, output="sos")
        if bandpass:
            self.bandpass_sos = butter(4, (low, high), btype="bandpass", fs=rate, output="sos")
        if notch:
            self.notch_coefficients = iirnotch(50.0, 30.0, fs=rate)

    def data_filter(self, rate, data, axis=-1):
        if self.settings is None or int(rate) != self.settings[0] or axis != -1:
            raise ValueError("滤波器尚未按当前采样率配置")
        block = np.asarray(data, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != 8:
            raise ValueError("滤波输入必须为 8 通道数组")
        if block.shape[1] == 0:
            return block.copy()

        if self.dc_sos is not None:
            if self.dc_state is None:
                self.dc_state = (
                    sosfilt_zi(self.dc_sos)[:, None, :] * block[:, 0][None, :, None]
                )
            block, self.dc_state = sosfilt(
                self.dc_sos, block, axis=-1, zi=self.dc_state
            )
        if self.bandpass_sos is not None:
            if self.bandpass_state is None:
                self.bandpass_state = (
                    sosfilt_zi(self.bandpass_sos)[:, None, :] * block[:, 0][None, :, None]
                )
            block, self.bandpass_state = sosfilt(
                self.bandpass_sos, block, axis=-1, zi=self.bandpass_state
            )
        if self.notch_coefficients is not None:
            b, a = self.notch_coefficients
            if self.notch_state is None:
                self.notch_state = lfilter_zi(b, a)[None, :] * block[:, 0, None]
            block, self.notch_state = lfilter(
                b, a, block, axis=-1, zi=self.notch_state
            )
        return block
