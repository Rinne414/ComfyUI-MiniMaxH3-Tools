"""Render the calibration CSV into the Markdown tables used by the report.

Kept separate from h3_bench.py so the tables can be regenerated (or extended
with someone else's GPU results) without re-running any benchmarks.

Usage:
  python h3_report.py [h3_calibration.csv] [vram_total_mib]
"""
import csv
import sys

DEFAULT_CSV = "h3_calibration.csv"
DEFAULT_VRAM_TOTAL_MIB = 24576  # RTX 3090 / 4090


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        for k in ("megapixels", "sit"):
            r[k] = float(r[k]) if r.get(k) else None
        for k in ("seconds", "steps", "width", "height", "frames", "latent_t",
                  "video_tokens", "peak_vram_mib"):
            r[k] = int(r[k]) if r.get(k) else None
        # overhead_s was added mid-sweep; earlier rows simply lack the column
        r["overhead_s"] = float(r["overhead_s"]) if r.get("overhead_s") else None
    return rows


def headroom(peak, vram_total):
    return vram_total - peak if peak else None


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    vram_total = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_VRAM_TOTAL_MIB
    rows = load(csv_path)
    ok = [r for r in rows if r["status"] == "success"]

    print("## Measured calibration — %d MiB VRAM\n" % vram_total)
    print("| MP | Resolution | Frames | Video tokens | s/it | Peak VRAM | Headroom | 20-step est. |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x["seconds"], x["megapixels"])):
        if r["status"] != "success":
            print("| %.1f | — | %ds | — | — | — | — | **%s** |"
                  % (r["megapixels"], r["seconds"], r["status"]))
            continue
        hr = headroom(r["peak_vram_mib"], vram_total)
        print("| %.1f | %dx%d | %d | %s | %.1f | %s MiB | %s MiB | %s min |"
              % (r["megapixels"], r["width"], r["height"], r["frames"],
                 format(r["video_tokens"], ","), r["sit"],
                 format(r["peak_vram_mib"], ","), format(hr, ","),
                 r["est_20step_min"] or "?"))

    print("\n## Cost scaling\n")
    print("| From | To | Tokens | s/it |")
    print("|---|---|---:|---:|")
    by_dur = {}
    for r in ok:
        by_dur.setdefault(r["seconds"], []).append(r)
    for dur, group in sorted(by_dur.items()):
        group.sort(key=lambda x: x["video_tokens"])
        for a, b in zip(group, group[1:]):
            print("| %.1f MP / %ds | %.1f MP / %ds | +%.0f%% | +%.0f%% |"
                  % (a["megapixels"], dur, b["megapixels"], dur,
                     100 * (b["video_tokens"] / a["video_tokens"] - 1),
                     100 * (b["sit"] / a["sit"] - 1)))
    # cross-duration at equal resolution
    mps = sorted({r["megapixels"] for r in ok})
    for mp in mps:
        same = sorted([r for r in ok if r["megapixels"] == mp],
                      key=lambda x: x["seconds"])
        for a, b in zip(same, same[1:]):
            print("| %.1f MP / %ds | %.1f MP / %ds | +%.0f%% | +%.0f%% |"
                  % (mp, a["seconds"], mp, b["seconds"],
                     100 * (b["video_tokens"] / a["video_tokens"] - 1),
                     100 * (b["sit"] / a["sit"] - 1)))

    if ok:
        print("\n## Per-token cost\n")
        print("| MP | Duration | ms per 1k tokens per step |")
        print("|---:|---:|---:|")
        for r in sorted(ok, key=lambda x: x["video_tokens"]):
            print("| %.1f | %ds | %.1f |"
                  % (r["megapixels"], r["seconds"],
                     1000 * r["sit"] / (r["video_tokens"] / 1000)))


if __name__ == "__main__":
    main()
