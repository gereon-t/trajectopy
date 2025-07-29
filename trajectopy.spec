from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []


tmp_ret_trajectopy = collect_all('trajectopy')
tmp_ret_rosbags = collect_all('rosbags')
tmp_ret_plotly = collect_all('plotly.graph_objs')


datas += tmp_ret_trajectopy[0] + tmp_ret_rosbags[0] + tmp_ret_plotly[0]
binaries += tmp_ret_trajectopy[1] + tmp_ret_rosbags[1] + tmp_ret_plotly[1]
hiddenimports += tmp_ret_trajectopy[2] + tmp_ret_rosbags[2] + tmp_ret_plotly[2] 


datas += tmp_ret_trajectopy[0]
binaries += tmp_ret_trajectopy[1]
hiddenimports += tmp_ret_trajectopy[2]

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
    optimize=0,
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
