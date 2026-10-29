# -*- mode: python ; coding: utf-8 -*-

import os

ICON_PATH = os.path.abspath(os.path.join(os.getcwd(), 'python_ads1299', 'mouse.ico'))

a = Analysis(
    ['main_uart_v1.2.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'serial',
        'serial.tools',
        'serial.tools.list_ports',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.ticker',
        'scipy',
        'scipy.signal',
        'fontTools',
        'pkg_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main_uart_v1.2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_PATH,
)
