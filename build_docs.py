"""Inject the generated calibration tables into README.md and the Civitai draft.

Both documents carry a placeholder comment; this replaces everything from the
placeholder up to the next `---` (README) or next `## ` heading (Civitai) so the
docs can be regenerated whenever the CSV grows.
"""
import io
import os
import runpy
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")
CIVITAI = os.path.join(HERE, "CIVITAI_DESCRIPTION.md")
CSV = os.path.join(HERE, "h3_calibration.csv")


def tables(vram_total="24576"):
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["h3_report.py", CSV, vram_total]
    try:
        with redirect_stdout(buf):
            runpy.run_path(os.path.join(HERE, "h3_report.py"), run_name="__main__")
    finally:
        sys.argv = argv
    return buf.getvalue().strip()


def splice(path, marker, end_pred, body):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    try:
        i = next(n for n, l in enumerate(lines) if marker in l)
    except StopIteration:
        print("!! marker %r not found in %s" % (marker, os.path.basename(path)))
        return False
    j = i + 1
    while j < len(lines) and not end_pred(lines[j]):
        j += 1
    out = lines[:i] + [marker, ""] + body.splitlines() + [""] + lines[j:]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("spliced %s (replaced lines %d-%d)" % (os.path.basename(path), i + 1, j))
    return True


def main():
    body = tables()
    splice(README, "<!-- CALIBRATION_TABLES -->",
           lambda l: l.strip() == "---", body)
    splice(CIVITAI, "<!-- CALIBRATION_PLACEHOLDER -->",
           lambda l: l.startswith("## "), body)


if __name__ == "__main__":
    main()
