"""Submit a saved API-format graph to ComfyUI and report the result.

Usage:
  python h3_run.py <api_graph.json> [megapixels] [seconds]

Environment:
  COMFY_SERVER   default http://127.0.0.1:8188
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")


def post(path, payload):
    req = urllib.request.Request(SERVER + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=60))


def main():
    path = sys.argv[1]
    graph = json.load(open(path, encoding="utf-8"))

    # allow overriding megapixels / seconds from argv for the resolution ladder
    if len(sys.argv) > 2:
        for nid, n in graph.items():
            if n["class_type"] == "ResolutionSelector":
                n["inputs"]["megapixels"] = float(sys.argv[2])
    if len(sys.argv) > 3:
        for nid, n in graph.items():
            if n["class_type"] == "PrimitiveFloat":
                n["inputs"]["value"] = float(sys.argv[3])

    try:
        res = post("/prompt", {"prompt": graph, "client_id": "h3_adv"})
    except urllib.error.HTTPError as e:
        print("SUBMIT REJECTED", e.code)
        print(e.read().decode()[:6000])
        return 1

    pid = res["prompt_id"]
    print("submitted %s  node_errors=%s" % (pid, res.get("node_errors")), flush=True)
    t0 = time.time()
    while True:
        time.sleep(15)
        el = (time.time() - t0) / 60
        hist = get("/history/" + pid)
        if pid in hist:
            h = hist[pid]
            st = h.get("status", {})
            print("\nDONE in %.1f min | %s" % (el, st.get("status_str")), flush=True)
            if st.get("status_str") != "success":
                for m in st.get("messages", [])[-15:]:
                    print("  ", str(m)[:800])
                return 1
            for nid, out in h.get("outputs", {}).items():
                for k, v in out.items():
                    print("  node %s %s: %s" % (nid, k, str(v)[:300]))
            return 0
        print("  t=%5.1f min" % el, flush=True)
        if el > 60:
            print("TIMEOUT")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
