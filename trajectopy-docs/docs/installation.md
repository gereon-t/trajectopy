# Installation

Trajectopy is available as a Python package and as a standalone app with the graphical user interface for
Windows, macOS and Linux. The app is made for the GUI: it needs no Python installation, but offers no Python
API, and the Windows app has no console output for the command line.

## Python Package

Trajectopy requires Python 3.10 or newer and is installed with pip. Choose the extras you need:

| Command | Contents |
| --- | --- |
| `pip install "trajectopy[gui]"` | Python package, plots and the GUI (recommended) |
| `pip install "trajectopy[plot]"` | Python package with plots ([`trajectopy.visualization`](api/visualization.md)), without the GUI |
| `pip install trajectopy` | Python package only: loading, alignment, evaluation and file I/O, no Qt |
| `pip install "trajectopy[fast]"` | Additionally reads large trajectory files several times faster (with [pyarrow](https://arrow.apache.org/docs/python/)) |

Extras can be combined, e.g. `pip install "trajectopy[gui,fast]"`.

The installation provides the `trajectopy` command, which starts the GUI and offers commands for the
[command line](user_guide/command_line.md).

To check the installation, print the versions of trajectopy and its dependencies:

```python
import trajectopy as tpy

tpy.show_versions()
```

Please add this output to bug reports.

### Updating

```console
pip install --upgrade "trajectopy[gui]"
```

Read the [release notes](release_notes.md) before updating to a new major version: they list the changes that
may require changes to your code.

## Apps

Every [release](https://github.com/gereon-t/trajectopy/releases/latest) provides

- `Trajectopy-<version>-setup.exe`: installer for Windows,
- `Trajectopy-<version>-macos-arm64.dmg`: app for macOS on Apple silicon,
- `Trajectopy-<version>-linux-x86_64.tar.gz`: app for Linux (x86-64).

The apps are not signed. On macOS, open the app the first time with right-click → *Open*.

### Windows

Download `Trajectopy-<version>-setup.exe` from the
[latest release](https://github.com/gereon-t/trajectopy/releases/latest) and run it. No Python installation or
administrator rights are needed: the installer installs for the current user, or for all users if you choose so
and confirm the administrator prompt. It adds Trajectopy to the Start menu and offers a desktop icon (off by
default) and to open `.traj` and `.tpy` files with Trajectopy on double-click (on by default). The app reports new versions (*Help → Check for Updates…*); installing
a newer version over an existing one updates it.

As the installer is not signed, Windows SmartScreen may warn about an unknown publisher: choose
*More info → Run anyway*.

## Development Installation

See [Contributing](contributing.md) for setting up a development environment.
