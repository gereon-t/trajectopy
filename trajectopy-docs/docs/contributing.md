# Contributing

Bug reports, ideas and code are welcome on [GitHub](https://github.com/gereon-t/trajectopy). Please add the
output of `trajectopy.show_versions()` to bug reports.

## Development Environment

Trajectopy uses [uv](https://docs.astral.sh/uv/) for its development environment:

```console
git clone https://github.com/gereon-t/trajectopy.git
cd trajectopy
uv sync --all-extras
uv run pre-commit install
```

The pre-commit hooks run before every commit: ruff (linting and formatting), the update of `uv.lock` and
`requirements.txt`, mypy and the tests.

## Tests

```console
uv run pytest
```

The tests cover the library, the GUI (without a display, with `QT_QPA_PLATFORM=offscreen`), the command line
and the documentation. Among others, they check that

- the public API does not change unintentionally (`test/test_public_api.py`),
- every public function, class, method and property has a complete docstring, every public name is in the API
  reference, every public function and class (except the exceptions) has examples, the examples of the
  documentation and of the docstrings run and show the actual output, and the scripts in `example_scripts`
  run (`test/test_docs.py`),
- trajectopy itself does not use its deprecated API (deprecation warnings are errors in the tests).

The CI checks formatting, linting and types, runs the tests on Linux and Windows with all supported Python
versions (with a minimum test coverage of 85 %) and once without the plotting and GUI extras, and builds the
documentation.

## Code Style

- Formatting and linting with ruff (line length 119), type checking with mypy.
- Docstrings in the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
  with `Args`, `Returns` and `Raises` sections, `Examples` that run as doctests (required for every public
  function and class) and, where helpful, `See Also` links written as ``[`trajectopy.ate`][trajectopy.ate]``.
- Changes of the public API follow the [API policy](api_policy.md): new options are keyword-only, functions do
  not modify their inputs, and nothing is removed without a deprecation (`trajectopy.utils.deprecation`).
- Every user-facing change is listed in `CHANGELOG.md`.

## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and
[mkdocstrings](https://mkdocstrings.github.io/) from the folder `trajectopy-docs`:

```console
cd trajectopy-docs
uv run mkdocs serve
```

The API reference is generated from the docstrings. Its pages in `docs/api` list the documented names; a new
public name has to be added there (the tests check this). Examples in the user guide run in a folder that
contains the example data of the repository (`example_data`) and a ROS2 bag (`recording.mcap`). An example that
cannot run there is marked with `<!-- skip-test: reason -->` in the line before it. Printed output shown in a
`text` block directly after an example is compared with the actual output.

`mkdocs build --strict` fails on broken links, anchors and references; the CI runs it for every change.
