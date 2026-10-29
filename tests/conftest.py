"""将自动记录设置与目录选择限制在测试临时目录，避免触碰用户配置。"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_recording_settings(tmp_path, monkeypatch):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QFileDialog

    previous_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    settings = QSettings("EEG_PET_DESK_ZC", "Recording")
    settings.clear()
    settings.sync()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: str(tmp_path / "records"))
    yield
    QSettings.setDefaultFormat(previous_format)
