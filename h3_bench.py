"""MiniMax-H3 calibration matrix for consumer GPUs.

Sweeps resolution x duration, recording steady-state s/it, peak VRAM, and
pass/OOM. Step count is held low because per-step VRAM and s/it do not depend
on the number of steps -- only wall-clock does, and that extrapolates linearly.

Results are appended to the CSV after every point so a crash or an OOM in the
middle still leaves usable data. Any point strictly harder than one that
already failed is skipped.

Usage:
  python h3_bench.py <api_graph.json> <comfyui_log_path> [steps]

Environment:
  COMFY_SERVER   default http://127.0.0.1:8188
  COMFY_OUTPUT   ComfyUI output dir; default <cwd>/output
  H3_CSV         result path; default ./h3_calibration.csv
  FFPROBE        ffprobe binary; default "ffprobe" from PATH
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")
OUT_CSV = os.environ.get("H3_CSV", "h3_calibration.csv")
OUTPUT_DIR = os.environ.get("COMFY_OUTPUT", os.path.join(os.getcwd(), "output"))
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"

# (megapixels, seconds) -- ordered cheapest first so an OOM ceiling shows up last
POINTS = [
    (0.4, 5), (0.6, 5), (0.8, 5), (1.0, 5),
    (0.4, 10), (0.6, 10), (0.8, 10), (1.0, 10),
]


def post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=60))


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


class VramWatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            self.peak = max(self.peak, gpu_used_mib())
            time.sleep(2)


def steady_sit(log_path, since_byte):
    """Parse tqdm 's/it' samples written after `since_byte`; return min (steady state)."""
    try:
        with open(log_path, "rb") as f:
            f.seek(since_byte)
            blob = f.read().decode("utf-8", "ignore")
    except Exception:
        return None, 0
    vals = [float(m) for m in re.findall(r"([\d.]+)s/it", blob)]
    return (min(vals) if vals else None), len(vals)


def probe(video_rel):
    p = os.path.join(OUTPUT_DIR, video_rel)
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames:format=duration", "-of", "csv=p=0", p],
            capture_output=True, text=True, timeout=30)
        parts = [x for x in out.stdout.replace("\n", ",").split(",") if x]
        return parts
    except Exception:
        return []


def latent_tokens(w, h, frames):
    lt = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return lt * (h // 32) * (w // 32), lt


def run_point(graph, mp, secs, steps, log_path):
    g = json.loads(json.dumps(graph))
    for n in g.values():
        ct = n["class_type"]
        if ct == "ResolutionSelector":
            n["inputs"]["megapixels"] = mp
        elif ct == "PrimitiveFloat":
            n["inputs"]["value"] = float(secs)
        elif ct == "BasicScheduler":
            n["inputs"]["steps"] = steps
        elif ct == "SaveVideo":
            n["inputs"]["filename_prefix"] = "bench/h3_%smp_%ss" % (mp, secs)

    log_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    watch = VramWatch()
    watch.start()
    t0 = time.time()
    row = {"megapixels": mp, "seconds": secs, "steps": steps}

    try:
        res = post("/prompt", {"prompt": g, "client_id": "h3_bench"})
    except urllib.error.HTTPError as e:
        watch.stop_flag = True
        row.update(status="submit_rejected", detail=e.read().decode()[:300])
        return row

    pid = res["prompt_id"]
    status, detail = "timeout", ""
    while time.time() - t0 < 60 * 60:
        time.sleep(5)
        hist = get("/history/" + pid)
        if pid in hist:
            h = hist[pid]
            st = h.get("status", {})
            status = st.get("status_str", "?")
            if status == "success":
                for out in h.get("outputs", {}).values():
                    for v in out.get("images", []):
                        if str(v.get("filename", "")).endswith(".mp4"):
                            row["video"] = os.path.join(v.get("subfolder", ""), v["filename"])
            else:
                detail = str(st.get("messages", ""))[-400:]
                low = detail.lower()
                if "out of memory" in low or "oom" in low or "alloc" in low:
                    status = "OOM"
            break

    watch.stop_flag = True
    elapsed = time.time() - t0
    sit, nsamp = steady_sit(log_path, log_size)

    row.update(status=status, wall_s=round(elapsed, 1),
               peak_vram_mib=watch.peak, sit=sit, sit_samples=nsamp,
               detail=detail[:200])

    if row.get("video"):
        p = probe(row["video"])
        if len(p) >= 2:
            w, hh = int(p[0]), int(p[1])
            dur = float(p[-1]) if p[-1].replace(".", "").isdigit() else 0.0
            frames = round(dur * 24)
            tok, lt = latent_tokens(w, hh, frames)
            row.update(width=w, height=hh, frames=frames, latent_t=lt, video_tokens=tok)
            if sit:
                # Everything outside the sampling loop -- weight load, text
                # encode, VAE decode -- is fixed cost that does not scale with
                # step count, so it carries over to the 20-step estimate as-is.
                overhead = elapsed - steps * sit
                row["overhead_s"] = round(overhead, 1)
                row["est_20step_min"] = round((overhead + 20 * sit) / 60, 1)
    return row


def main():
    graph = json.load(open(sys.argv[1], encoding="utf-8"))
    log_path = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    cols = ["megapixels", "seconds", "steps", "status", "width", "height", "frames",
            "latent_t", "video_tokens", "sit", "sit_samples", "peak_vram_mib",
            "wall_s", "overhead_s", "est_20step_min", "video", "detail"]
    new = not os.path.exists(OUT_CSV)
    fh = open(OUT_CSV, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fh, fieldnames=cols)
    if new:
        wr.writeheader()
        fh.flush()

    # A point is strictly harder than another when both its resolution and its
    # duration are >=, so anything at or above a known failure can be skipped.
    failed = []
    for mp, secs in POINTS:
        if any(fmp <= mp and fsec <= secs for fmp, fsec in failed):
            print("skip %.1f MP / %ds  (a strictly easier point already failed)"
                  % (mp, secs), flush=True)
            continue
        print("\n>>> %.1f MP / %ds / %d steps" % (mp, secs, steps), flush=True)
        row = run_point(graph, mp, secs, steps, log_path)
        print("    %s  %sx%s  tok=%s  s/it=%s  peakVRAM=%s MiB  wall=%.1f min"
              % (row.get("status"), row.get("width"), row.get("height"),
                 row.get("video_tokens"), row.get("sit"), row.get("peak_vram_mib"),
                 (row.get("wall_s") or 0) / 60), flush=True)
        if row.get("detail"):
            print("    detail:", row["detail"][:200], flush=True)
        wr.writerow({k: row.get(k, "") for k in cols})
        fh.flush()
        if row.get("status") != "success":
            failed.append((mp, secs))
    fh.close()
    print("\nwrote", OUT_CSV)


if __name__ == "__main__":
    main()
