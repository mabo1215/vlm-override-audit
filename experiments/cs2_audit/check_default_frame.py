"""Quantify the default-box coordinate-frame error in the audited generation logs.

Runs read-only next to the frozen logs.  For every generation-log row it
re-parses the VLM's raw direct answer exactly as the generator did, then
normalizes the pixel box twice: by the original image size (what the
generator did) and by the size of the resized image the VLM actually saw
(Qwen2.5-VL smart resize).  It also records the candidate-set composition and
the scoring-log view of the default box.  Output: one gzipped CSV on stdout; settings are echoed to stderr.

Example:
    python check_default_frame.py --logs-root <results dir> --weights <checkpoint> > per_row.csv.gz
"""

import glob
import gzip
import io
import json
import math
import re
import sys

import pandas as pd

ROOT = ""  # set by --logs-root: directory holding cs2_scaleup_<benchmark>/{candidates,scores,*.parquet}
WEIGHTS = ""  # set by --weights: the Qwen2.5-VL-7B-Instruct checkpoint used for generation
BENCHMARKS = {"RefCOCO+": "refcocoplus", "RefCOCOg": "refcocog", "RefCOCO": "refcoco"}
MAX_PIXELS = 1_003_520  # hard-coded in the generator's processor call

BBOX_RE = re.compile(r"\"bbox_2d\"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")
FOUR_RE = re.compile(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)")


def preprocessor_settings():
    try:
        cfg = json.load(open(f"{WEIGHTS}/preprocessor_config.json"))
    except OSError:
        cfg = {}
    patch = int(cfg.get("patch_size", 14))
    merge = int(cfg.get("merge_size", 2))
    min_pixels = int(cfg.get("min_pixels", 56 * 56))
    return patch * merge, min_pixels, cfg


def smart_resize(height, width, factor, min_pixels, max_pixels):
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ground_truth(paths):
    truth = {}
    for path in paths:
        frame = pd.read_parquet(path, columns=["ref_id", "bbox", "raw_image_info"])
        for _, row in frame.iterrows():
            info = json.loads(row["raw_image_info"])
            w, h = float(info["width"]), float(info["height"])
            x1, y1, x2, y2 = (float(v) for v in row["bbox"])
            truth.setdefault(str(row["ref_id"]), ((x1 / w, y1 / h, x2 / w, y2 / h), w, h))
    return truth


def main():
    import argparse

    global ROOT, WEIGHTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    ROOT, WEIGHTS = args.logs_root.rstrip("/"), args.weights.rstrip("/")
    factor, min_pixels, cfg = preprocessor_settings()
    sys.stderr.write(json.dumps({"factor": factor, "min_pixels": min_pixels,
                                 "preprocessor_keys": sorted(cfg)}) + "\n")
    try:
        gen_cfg = json.load(open(f"{WEIGHTS}/generation_config.json"))
        sys.stderr.write("generation_config " + json.dumps(gen_cfg) + "\n")
    except OSError:
        sys.stderr.write("generation_config missing\n")
    out = []
    for bench, tag in BENCHMARKS.items():
        base = f"{ROOT}/cs2_scaleup_{tag}"
        truth = ground_truth(sorted(glob.glob(f"{base}/{tag}_scaleup_*.parquet")))
        scores = {}
        meta = {}
        for path in sorted(glob.glob(f"{base}/scores/*_scores.json")):
            payload = json.load(open(path))
            meta.setdefault("permutations", payload.get("permutations"))
            for entry in payload.get("scores", []):
                scores[str(entry["example_id"])] = entry
        sys.stderr.write(f"{bench} score meta {meta}\n")
        for path in sorted(glob.glob(f"{base}/candidates/*.json")):
            phase = "confirmation" if "confirmation" in path else "calibration"
            for row in json.load(open(path)).get("logs", []):
                if "error" in row:
                    out.append({"benchmark": bench, "phase": phase, "example_id": str(row.get("example_id")),
                                "status": "error_row"})
                    continue
                ex = str(row["example_id"])
                gt, w, h = truth.get(ex, (None, None, None))
                text = row.get("restricted_text", "")
                match = BBOX_RE.search(text) or FOUR_RE.search(text)
                rec = {"benchmark": bench, "phase": phase, "example_id": ex, "image_id": row.get("image_id"),
                       "width": w, "height": h, "logged_iou": row.get("restricted_iou"),
                       "gen_sources": "|".join(row.get("candidate_sources", [])),
                       "gen_num_candidates": row.get("num_candidates")}
                entry = scores.get(ex)
                if entry is not None:
                    src = list(entry["candidate_sources"])
                    rec["score_num_candidates"] = len(src)
                    rec["score_has_restricted"] = "restricted" in src
                    if "restricted" in src:
                        k = src.index("restricted")
                        rec["score_restricted_index"] = k
                        sbox = entry["candidate_boxes"][k]
                        rec["score_restricted_iou"] = iou(sbox, gt) if gt else None
                        rec["score_restricted_matches_gen"] = (
                            max(abs(a - b) for a, b in zip(sbox, row["restricted_box"])) < 1e-9)
                if gt is None or match is None:
                    rec["status"] = "no_gt" if gt is None else "parse_failure"
                    out.append(rec)
                    continue
                px = [float(v) for v in match.groups()]
                hr, wr = smart_resize(h, w, factor, min_pixels, MAX_PIXELS)
                rec.update({"resized_width": wr, "resized_height": hr, "px_x1": px[0], "px_y1": px[1],
                            "px_x2": px[2], "px_y2": px[3]})
                if max(px) > 1.0:
                    used = (px[0] / w, px[1] / h, px[2] / w, px[3] / h)
                    fixed = (px[0] / wr, px[1] / hr, px[2] / wr, px[3] / hr)
                    fixed = tuple(min(1.0, max(0.0, v)) for v in fixed)
                    rec["status"] = "pixel"
                else:
                    used = fixed = tuple(px)
                    rec["status"] = "normalized"
                rec["used_iou"] = iou(used, gt)
                rec["fixed_iou"] = iou(fixed, gt)
                rec["out_of_range"] = max(used) > 1.0
                rec["exceeds_resized"] = px[2] > wr or px[3] > hr
                out.append(rec)
    frame = pd.DataFrame(out)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(frame.to_csv(index=False).encode())
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
