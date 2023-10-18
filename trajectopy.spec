# -*- mode: python ; coding: utf-8 -*-

added_files = [
    ( 'trajectopy/version', 'trajectopy/' ),
    ( 'trajectopy/default.mplstyle', 'trajectopy/' ),
    ( 'trajectopy/resources/icon.png', 'trajectopy/resources/'),
    ( 'trajectopy/resources/full-icon-poppins.png', 'trajectopy/resources/')
]

a = Analysis(
    ['trajectopy\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=['trajectopy_core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='trajectopy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
