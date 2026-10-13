from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []


tmp_ret_trajectopy = collect_all('trajectopy')
tmp_ret_rosbags = collect_all('rosbags')
# pyqtgraph ships its colormaps as data files and imports its OpenGL items dynamically
tmp_ret_pyqtgraph = collect_all('pyqtgraph')

for tmp_ret in (tmp_ret_trajectopy, tmp_ret_rosbags, tmp_ret_pyqtgraph):
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['trajectopy\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
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
    icon='trajectopy\\gui\\resources\\icon-bg.ico',
)
