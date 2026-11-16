"""The README's examples, checked and their plots regenerated: README.md
is the single source of them.

    pytest tests/test_readme.py     # check the README
    python tests/test_readme.py     # regenerate its plots (plots/*.svg)

The Python code blocks run in order, in one namespace (in a temporary
folder when testing, so files they write don't land in the repo).
Blocks after a line <!-- skip-test --> (invisible on GitHub) don't run:
API summaries with names defined elsewhere.

Printed output: the comment lines right after a statement that prints
(e.g. print(f) or model.describe()) are its expected output, checked
line by line (whitespace normalized, numbers to a relative tolerance,
other printed lines, e.g. IPOPT's banner, ignored). A comment after a
statement that prints nothing is just a comment.

Plots: plt.show() closes the figure; when regenerating it saves it to
the SVG of the image link after the block (![...](plots/name.svg)),
without a date and with fixed element ids, so a file only changes when
its plot does.
"""

import ast
import contextlib
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

README = Path(__file__).resolve().parent.parent / "README.md"
SKIP = "<!-- skip-test -->"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+\.svg)\)")
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
RTOL = 1e-6


def readme_blocks(text):
    """(line, code, svg) of each Python block to run: its first code
    line's number in the README, its code, and the SVG path of the image
    link between it and the next block (or None)."""
    matches = list(BLOCK.finditer(text))
    blocks = []
    for i, m in enumerate(matches):
        before = text[: m.start()].rstrip().splitlines()
        if before and before[-1].strip() == SKIP:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        image = IMAGE.search(text, m.end(), end)
        line = text[: m.start()].count("\n") + 2
        blocks.append((line, m.group(1), image and image.group(1)))
    return blocks


def _expected_output(lines, end):
    """The comment lines right after line end (1-based, in lines),
    without their '#', up to a blank or code line."""
    expected = []
    for line in lines[end:]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        expected.append(stripped[1:].strip())
    return expected


def _lines_match(expected, got):
    """Equal apart from whitespace and numbers within RTOL."""
    expected, got = " ".join(expected.split()), " ".join(got.split())
    if NUMBER.sub("#", expected) != NUMBER.sub("#", got):
        return False
    return all(
        abs(float(a) - float(b)) <= RTOL * max(abs(float(a)), abs(float(b)))
        for a, b in zip(NUMBER.findall(expected), NUMBER.findall(got))
    )


def _output_problem(expected, printed):
    """None if the expected lines are among the printed ones, in order;
    else a description."""
    got = [line for line in printed.splitlines() if line.strip()]
    i = 0
    for line in expected:
        while i < len(got) and not _lines_match(line, got[i]):
            i += 1
        if i == len(got):
            return f"expected {line!r}, printed:\n" + "\n".join(got)
        i += 1
    return None


def run_readme(save_plots=False, root=README.parent):
    """Run the README's blocks; a list of the output mismatches (with the
    README line). save_plots: save each block's figure to its SVG (paths
    relative to root). Exceptions propagate, with the README line."""
    namespace = {"__name__": "__readme__"}
    problems = []
    svg_path = None

    def show(*args, **kwargs):
        if save_plots and svg_path is not None:
            matplotlib.rcParams["svg.hashsalt"] = "cas_models"
            plt.gcf().savefig(root / svg_path, metadata={"Date": None})
            print(f"saved {root / svg_path}", file=sys.__stdout__)
        plt.close("all")

    original_show, plt.show = plt.show, show
    try:
        for first_line, code, svg_path in readme_blocks(README.read_text()):
            lines = code.splitlines()
            for stmt in ast.parse(code).body:
                where = f"README.md:{first_line + stmt.lineno - 1}"
                buffer = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buffer):
                        exec(
                            compile(ast.Module([stmt], []), where, "exec"),
                            namespace,
                        )
                except Exception as err:
                    raise RuntimeError(f"{where}: {err!r}") from err
                expected = _expected_output(lines, stmt.end_lineno)
                if expected and buffer.getvalue().strip():
                    problem = _output_problem(expected, buffer.getvalue())
                    if problem:
                        problems.append(f"{where}: {problem}")
    finally:
        plt.show = original_show
        plt.close("all")
    return problems


def test_readme(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    problems = run_readme()
    assert not problems, "\n\n".join(problems)


if __name__ == "__main__":
    for problem in run_readme(save_plots=True):
        print(problem, file=sys.stderr)
