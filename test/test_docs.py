"""Keeps the documentation in line with the code.

- Every Python example of the documentation runs (with the example data of the repository), and printed output
  shown in a ``text`` block right after an example matches the actual output. The examples find the example data
  of the repository in ``example_data`` and a ROS2 bag in ``recording.mcap``. Examples preceded by a
  ``<!-- skip-test: reason -->`` comment are not run (e.g. they need an internet connection).
- The scripts in ``example_scripts`` run.
- The ``Examples`` sections of the docstrings run and show the actual output (doctests).
- Every public name is documented in the API reference.
- Every public function, class, method and property has a complete docstring (arguments and return value).
"""

import contextlib
import doctest
import enum
import importlib
import inspect
import io
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.test_public_api import PUBLIC_API, importable_namespaces, plotting_available

REPOSITORY = Path(__file__).resolve().parent.parent
DOCS = REPOSITORY / "trajectopy-docs" / "docs"
GUIDE_PAGES = sorted(path for path in DOCS.rglob("*.md") if "api" not in path.relative_to(DOCS).parts)

_BLOCK = re.compile(r"(?P<skip><!-- skip-test[^>]*-->\s*\n)?```(?P<lang>\w*)[^\n]*\n(?P<code>.*?)\n```", re.S)


def _examples(page: Path) -> list[tuple[str, str | None, bool]]:
    """Python examples of a page: (code, expected output or None, skipped)."""
    text = page.read_text(encoding="utf-8")
    blocks = list(_BLOCK.finditer(text))
    examples = []
    for index, block in enumerate(blocks):
        if block.group("lang") != "python":
            continue
        expected = None
        if index + 1 < len(blocks):
            following = blocks[index + 1]
            between = text[block.end() : following.start()]
            if following.group("lang") == "text" and not between.strip():
                expected = following.group("code")
        examples.append((block.group("code"), expected, block.group("skip") is not None))
    return examples


def _write_recording(directory: Path) -> None:
    """Writes ``recording.mcap``, the ROS2 bag of the examples, with the topics /odom, /tf, /fix and /path."""
    from test.test_rosbag import write_bag

    bag = write_bag(directory / "bag")
    shutil.move(next(bag.glob("*.mcap")), directory / "recording.mcap")
    shutil.rmtree(bag)


@contextlib.contextmanager
def _without_windows():
    """Replaces ``trajectopy.visualization.show``, which would open a window and block."""
    if not plotting_available():
        yield
        return
    with mock.patch("trajectopy.visualization.show"):
        yield


class TestDocumentationExamples(unittest.TestCase):
    def test_examples_run(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        for page in GUIDE_PAGES:
            examples = _examples(page)
            if not examples or (not plotting_available() and any("visualization" in e[0] for e in examples)):
                continue  # pages with plots need the plotting extras
            with self.subTest(page=str(page.relative_to(DOCS))), tempfile.TemporaryDirectory() as directory:
                shutil.copytree(REPOSITORY / "example_data", Path(directory) / "example_data")
                _write_recording(Path(directory))
                namespace: dict = {"__name__": "__docs__"}
                cwd = os.getcwd()
                os.chdir(directory)
                try:
                    for number, (code, expected, skipped) in enumerate(examples, start=1):
                        if skipped:
                            continue
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output), _without_windows():
                            try:
                                exec(compile(code, f"{page.name} example {number}", "exec"), namespace)  # noqa: S102
                            except Exception as error:
                                self.fail(f"{page.relative_to(DOCS)}, example {number} failed: {error!r}\n{code}")
                        if expected is not None:
                            self.assertEqual(
                                output.getvalue().strip(),
                                expected.strip(),
                                f"{page.relative_to(DOCS)}, example {number}: the shown output is outdated",
                            )
                finally:
                    os.chdir(cwd)


class TestExampleScripts(unittest.TestCase):
    def test_example_scripts_run(self):
        """The scripts in ``example_scripts`` run with the example data (they plot, so they need the extras)."""
        if not plotting_available():
            self.skipTest("plotting dependencies not installed")
        import runpy

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        scripts = sorted((REPOSITORY / "example_scripts").glob("*.py"))
        self.assertTrue(scripts)
        with tempfile.TemporaryDirectory() as directory:
            shutil.copytree(REPOSITORY / "example_data", Path(directory) / "example_data")
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                for script in scripts:
                    with (
                        self.subTest(script=script.name),
                        _without_windows(),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        runpy.run_path(str(script), run_name="__main__")
            finally:
                os.chdir(cwd)


class TestDocstringExamples(unittest.TestCase):
    """The ``Examples`` sections of the docstrings of the public API run and show the actual output."""

    def test_docstring_examples(self):
        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        tests: dict[str, doctest.DocTest] = {}
        for namespace in importable_namespaces():
            module = importlib.import_module(namespace)
            for name in PUBLIC_API[namespace]:
                obj = getattr(module, name)
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    for test in finder.find(obj, f"{namespace}.{name}", globs={}):
                        if test.examples:
                            tests.setdefault(test.name, test)
        self.assertGreater(len(tests), 100)

        with tempfile.TemporaryDirectory() as directory:
            shutil.copytree(REPOSITORY / "example_data", Path(directory) / "example_data")
            _write_recording(Path(directory))
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                output = io.StringIO()
                for test in tests.values():
                    runner.run(test, out=output.write)
            finally:
                os.chdir(cwd)
        results = runner.summarize(verbose=False)
        self.assertEqual(results.failed, 0, output.getvalue())

    def test_every_public_name_has_examples(self):
        """Every public function and class shows its use (exceptions excepted)."""
        missing = []
        for namespace in importable_namespaces():
            module = importlib.import_module(namespace)
            for name in PUBLIC_API[namespace]:
                obj = getattr(module, name)
                if name.startswith("__") or (inspect.isclass(obj) and issubclass(obj, BaseException)):
                    continue
                if "Examples:" not in (inspect.getdoc(obj) or ""):
                    missing.append(f"{namespace}.{name}")
        self.assertEqual(missing, [], "Public names without an Examples section")


class TestApiReference(unittest.TestCase):
    def test_every_public_name_is_documented(self):
        """Every public object is documented, under one of its public names (e.g. exceptions are available from
        ``trajectopy`` and ``trajectopy.exceptions``)."""
        reference = "\n".join(path.read_text(encoding="utf-8") for path in (DOCS / "api").glob("*.md"))
        namespaces = importable_namespaces()
        documented = set()
        for qualname in re.findall(r"^::: ([\w.]+)", reference, re.M):
            namespace, _, name = qualname.rpartition(".")
            if namespace in namespaces:
                documented.add(id(getattr(importlib.import_module(namespace), name)))
        missing = [
            f"{namespace}.{name}"
            for namespace in namespaces
            for name in PUBLIC_API[namespace]
            if name != "__version__" and id(getattr(importlib.import_module(namespace), name)) not in documented
        ]
        self.assertEqual(missing, [], "Public names missing in the API reference")

    def test_documented_names_are_public(self):
        reference = "\n".join(path.read_text(encoding="utf-8") for path in (DOCS / "api").glob("*.md"))
        public = {f"{namespace}.{name}" for namespace, names in PUBLIC_API.items() for name in names}
        internal = [name for name in re.findall(r"^::: ([\w.]+)", reference, re.M) if name not in public]
        self.assertEqual(internal, [], "The API reference documents names that are not public")


def _documented_parameters(doc: str) -> set[str]:
    names: set[str] = set()
    for section in re.finditer(r"^(?:Args|Attributes):\n(.*?)(?=^\S|\Z)", doc, re.M | re.S):
        names |= set(re.findall(r"^\s{4}\**(\w+)\s*(?:\(|:)", section.group(1), re.M))
    return names


class TestDocstrings(unittest.TestCase):
    def _check_function(self, qualname: str, function, class_doc: str = "") -> list[str]:
        doc = inspect.getdoc(function) or ""
        if not doc:
            return [f"{qualname}: no docstring"]
        problems = []
        signature = inspect.signature(function)
        parameters = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.name not in ("self", "cls")
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        documented = _documented_parameters(doc) | _documented_parameters(class_doc)
        if missing := [name for name in parameters if name not in documented]:
            problems.append(f"{qualname}: undocumented arguments {missing}")
        if signature.return_annotation not in (inspect.Signature.empty, None, "None") and "Returns:" not in doc:
            problems.append(f"{qualname}: no Returns section")
        return problems

    def _check_class(self, qualname: str, cls) -> list[str]:
        class_doc = inspect.getdoc(cls) or ""
        if not class_doc:
            return [f"{qualname}: no docstring"]
        if issubclass(cls, (BaseException, enum.Enum)):
            return []
        problems = []
        for name, member in vars(cls).items():
            if name.startswith("_") and name != "__init__":
                continue
            target = f"{qualname}.{name}"
            if isinstance(member, property):
                if not (member.fget and inspect.getdoc(member.fget)):
                    problems.append(f"{target}: no docstring")
            elif isinstance(member, (staticmethod, classmethod)):
                problems += self._check_function(target, member.__func__)
            elif inspect.isfunction(member):
                if name == "__init__":
                    own = inspect.cleandoc(member.__doc__ or "")
                    parameters = [p for p in inspect.signature(member).parameters if p != "self"]
                    documented = _documented_parameters(own) | _documented_parameters(class_doc)
                    if missing := [p for p in parameters if p not in documented]:
                        problems.append(f"{target}: undocumented arguments {missing}")
                else:
                    problems += self._check_function(target, member)
        return problems

    def test_public_docstrings_are_complete(self):
        problems: list[str] = []
        seen: set[int] = set()
        for namespace in importable_namespaces():
            module = importlib.import_module(namespace)
            for name in PUBLIC_API[namespace]:
                obj = getattr(module, name)
                if name.startswith("__") or id(obj) in seen:
                    continue
                seen.add(id(obj))
                if inspect.isclass(obj):
                    problems += self._check_class(f"{namespace}.{name}", obj)
                elif callable(obj):
                    problems += self._check_function(f"{namespace}.{name}", obj)
        self.assertEqual(problems, [], "Incomplete docstrings:\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
