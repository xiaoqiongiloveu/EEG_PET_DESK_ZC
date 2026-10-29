import os
import sys
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

ROOT = r'E:\CW-EEG\python_ads1299'
PKG = os.path.join(ROOT, 'python_ads1299')
for p in [ROOT, PKG]:
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import QTimer

import importlib.util

spec = importlib.util.spec_from_file_location('main_uart_v1_2', os.path.join(PKG, 'main_uart_v1.2.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

app = QApplication([])
win = mod.MainWindow()
win.resize(1280, 760)
win.show()

# Give the UI time to render before capturing.
QTimer.singleShot(800, lambda: save_and_quit())


def save_and_quit():
    shot = QGuiApplication.primaryScreen().grabWindow(win.winId())
    out = os.path.join(ROOT, 'ui_preview.png')
    ok = shot.save(out, 'png')
    print('SAVED', out, ok)
    app.quit()

app.exec_()
