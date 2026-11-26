"""Prints the section of a version in CHANGELOG.md as text for its GitHub release.

The changelog is wrapped at 119 characters, which Markdown joins again, but GitHub shows every line break of a
release text. So the lines of each paragraph and list item are joined; headings, tables and code blocks are kept.

Usage: python .github/release_notes.py 7.0.0 [CHANGELOG.md]
"""

import re
import sys

# lines that start a block of their own: headings, tables, code fences and list items (also nested ones)
STARTS_BLOCK = re.compile(r"^(#|\||```|\s*[-*] |\s*\d+\. )")
# lines that end a block: headings, table rows and bold lines used as subheadings
ENDS_BLOCK = re.compile(r"^(#|\||```|\*\*[^*].*\*\*\s*$)")


def section(text: str, version: str) -> list[str]:
    """The lines of the section ``## <version>`` without its heading."""
    lines = text.splitlines()
    heading = f"## {version}"
    if heading not in lines:
        raise SystemExit(f"CHANGELOG.md has no section '{heading}'")
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def unwrap(lines: list[str]) -> str:
    """Joins the wrapped lines of paragraphs and list items."""
    out: list[str] = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        previous = out[-1] if out else ""
        continues = (
            not in_code
            and line.strip()
            and previous.strip()
            and not STARTS_BLOCK.match(line)
            and not ENDS_BLOCK.match(previous)
        )
        if continues:
            out[-1] = f"{previous.rstrip()} {line.strip()}"
        else:
            out.append(line)
    return "\n".join(out).strip() + "\n"


if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) > 2 else "CHANGELOG.md"
    with open(path, encoding="utf-8") as file:
        print(unwrap(section(file.read(), sys.argv[1])), end="")
