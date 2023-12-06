# -*- mode: python ; coding: utf-8 -*-

added_files = [
    ( 'trajectopy/version', 'trajectopy/' ),
    ( 'trajectopy/resources/icon.png', 'trajectopy/resources/'),
    ( 'trajectopy/resources/icon-bg.png', 'trajectopy/resources/'),
    ( 'trajectopy/resources/full-icon-poppins.png', 'trajectopy/resources/'),
    ( 'reportdata/generic.html', 'trajectopy/templates/'),
    ( 'reportdata/multi_template.html', 'trajectopy/templates/'),
    ( 'reportdata/single_template.html', 'trajectopy/templates/'),
    ( 'reportdata/icon.png', 'trajectopy/assets/'),
    ( 'reportdata/igg.png', 'trajectopy/assets/'),
    ( 'reportdata/uni_bonn.png', 'trajectopy/assets/'),
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
    name='Trajectopy',
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
    icon="trajectopy\\resources\\icon-bg.png"
)
