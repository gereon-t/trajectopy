# Installation

Trajectopy is available as a Python package and, for Windows, as a standalone app with the graphical user
interface. Both contain the same functionality.

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

## Windows App

Download `Trajectopy-<version>-setup.exe` from the
[latest release](https://github.com/gereon-t/trajectopy/releases/latest) and run it. No Python installation or
administrator rights are needed. The installer adds Trajectopy to the Start menu and, optionally, opens `.traj`
and `.tpy` files with it on double-click. The app reports new versions (*Help → Check for Updates…*); installing
a newer version over an existing one updates it.

As the installer is not signed, Windows SmartScreen may warn about an unknown publisher: choose
*More info → Run anyway*.

## Development Installation

See [Contributing](contributing.md) for setting up a development environment.
