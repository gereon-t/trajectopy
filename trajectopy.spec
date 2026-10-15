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
    # optional speed-up for large files (trajectopy[fast]); kept out of the app, it adds ~90 MB
    excludes=["pyarrow"],
    noarchive=False,
    optimize=1,
)
# Python's ssl module (update check) needs the OpenSSL DLLs shipped with this Python. PyInstaller may pick
# same-named DLLs of another OpenSSL version from PATH instead (e.g. Git for Windows' when building in Git
# Bash), and then ssl cannot be loaded in the app.
import os
import _ssl

_python_dlls = os.path.dirname(_ssl.__file__)
a.binaries = [
    (name, os.path.join(_python_dlls, os.path.basename(name)), typecode)
    if os.path.basename(name).lower().startswith(('libcrypto-', 'libssl-'))
    and os.path.isfile(os.path.join(_python_dlls, os.path.basename(name)))
    else (name, source, typecode)
    for name, source, typecode in a.binaries
]

pyz = PYZ(a.pure)

# One-folder build (dist/Trajectopy/): starts much faster than a one-file exe, which unpacks itself on every
# start. installer/trajectopy.iss packages the folder into a Windows installer.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Trajectopy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='trajectopy\\gui\\resources\\icon-bg.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Trajectopy',
)
