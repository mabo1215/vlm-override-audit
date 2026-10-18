#!/usr/bin/env bash
# Driver for the GPU arms G1-G3 declared in docs/Design.md section 15.
# Runs on the vGPU 3090 or on the lab host 4c (4x RTX 3090 24 GB, shared
# with other users; only the GPUs listed in GPUS are used).  Each step is
# idempotent and resumable:
#
#   bash experiments/cs2_audit/vgpu_g_arms.sh preflight
#   bash experiments/cs2_audit/vgpu_g_arms.sh setup        # venv (no GPU needed)
#   bash experiments/cs2_audit/vgpu_g_arms.sh download     # weights and data (no GPU needed)
#   bash experiments/cs2_audit/vgpu_g_arms.sh inputs       # label-free records + separate GT
#   bash experiments/cs2_audit/vgpu_g_arms.sh protocol     # protocol file; commit it before any label join
#   bash experiments/cs2_audit/vgpu_g_arms.sh pilot        # 60 training records, GPU
#   bash experiments/cs2_audit/vgpu_g_arms.sh generate ARM WORKER NWORKERS   # ARM in g1|g2|g3
#   bash experiments/cs2_audit/vgpu_g_arms.sh sysa         # System A's archived candidate sets as records (CPU)
#   bash experiments/cs2_audit/vgpu_g_arms.sh features ARM                  # V1 pair features, GPU; ARM in g1|g2|g3|sysa
#   bash experiments/cs2_audit/vgpu_g_arms.sh queue WORKER NWORKERS         # after G1: G2 then G3 on this GPU
#   bash experiments/cs2_audit/vgpu_g_arms.sh feature_queue NWORKERS        # V1 features on GPU 2 next to generation
#   bash experiments/cs2_audit/vgpu_g_arms.sh requeue WORKER NWORKERS       # swap a queue screen once its G2 shard ends
#
# Worker i runs on the i-th GPU in GPUS.  A step refuses a GPU that runs
# another user's job or lacks the memory it needs (generation 20,000 MiB,
# features 3,500 MiB), so one feature run can share a GPU with one
# generation run.  A lock per output file stops a step from running twice.
#
# No step reads ground truth except the separate label joins, which are run
# by hand in the protocol's order (join_system_b_labels.py).
set -euo pipefail

case "$(hostname)" in
  ubuntu-4card)  # lab host 4c
    BASE="${BASE:-/data1/mabo1215}"
    : "${COCO:=/data1/datasets/refcoco_family_images/train2014}"
    : "${SYSA_LOGS:=${BASE}/system_a_logs}"
    : "${EXTRA_EXCLUDE:=${BASE}/system_a_logs/**/*.parquet}"
    : "${GPUS:=2,3}"
    : "${HF_HOME:=${BASE}/hf_home}"
    ;;
  *)  # vGPU 3090
    BASE="${BASE:-/root/autodl-tmp}"
    : "${DINO:=${BASE}/HKVLM/data/weights/grounding-dino-base}"
    : "${COCO:=${BASE}/coco_images/train2014}"
    : "${SYSA_LOGS:=${BASE}/BodhiNet_runs/BodhiNet_cs2_scaleup/src/experiments/vlm_self_distillation/results}"
    : "${EXTRA_EXCLUDE:=${BASE}/BodhiNet*/**/results/**/*.parquet}"
    : "${GPUS:=0}"
    ;;
esac
REPO="${REPO:-${BASE}/wacv2027_g}"
WORK="${WORK:-${BASE}/wacv2027_garms}"
VENV="${VENV:-${BASE}/venv_wacv2027_garms}"
WEIGHTS="${WEIGHTS:-${BASE}/weights_wacv2027}"
DINO="${DINO:-${WEIGHTS}/grounding-dino-base}"
PY="${VENV}/bin/python"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${HF_HOME:-}" ]]; then export HF_HOME; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "${REPO}"
mkdir -p "${WORK}/logs"

preflight() {
  echo "commit: $(git rev-parse HEAD)  (must equal the pushed local commit)"
  git status --short | head -5
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || echo "no GPU visible (no-card mode)"
  echo "GPUs for this run: ${GPUS}"
  df -h "${BASE}" | tail -1
  df -i "${BASE}" | tail -1
  for p in "${DINO}" "${COCO}" "${SYSA_LOGS}"; do [[ -e "${p}" ]] && echo "ok  ${p}" || echo "MISSING ${p}"; done
}

setup() {
  [[ -x "${PY}" ]] || python3 -m venv "${VENV}"
  # PyPI's Linux torch wheels already bundle CUDA 12.8; no cache keeps the small root disk free.
  local pip=("${PY}" -m pip install -q --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple)
  "${pip[@]}" --upgrade pip
  "${pip[@]}" "torch==2.8.0" "torchvision==0.23.0"
  # sentencepiece and protobuf: SigLIP's tokenizer (V1 features)
  "${pip[@]}" "transformers>=4.52,<5" accelerate pandas pyarrow scipy scikit-learn pillow huggingface_hub timm sentencepiece protobuf
  "${PY}" -c "import torch, transformers, sklearn, sentencepiece; print(torch.__version__, transformers.__version__, sklearn.__version__)"
}

hf_get() {  # hf_get REPO_ID LOCAL_DIR [REPO_TYPE [ALLOW_PATTERN ...]]
  "${PY}" - "$@" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, local, *rest = sys.argv[1:]
kind = rest[0] if rest else "model"
allow = rest[1:] or None
path = snapshot_download(repo, repo_type=kind, local_dir=local, allow_patterns=allow,
                         ignore_patterns=["*.msgpack", "*.h5", "*onnx*", "*.ot"])
print("downloaded", repo, "->", path, flush=True)
PYEOF
}

download() {
  mkdir -p "${WEIGHTS}" "${WORK}/data"
  for repo in OpenGVLab/InternVL3-8B-hf google/siglip-so400m-patch14-384 openai/clip-vit-large-patch14-336; do
    hf_get "${repo}" "${WEIGHTS}/${repo##*/}"
  done
  [[ -e "${DINO}/config.json" ]] || hf_get IDEA-Research/grounding-dino-base "${DINO}"
  hf_get jxu124/refclef-benchmark "${WORK}/data/refclef" dataset "data/refclef_berkeley_test-*"
  # All splits: train for the verifier pool, the rest to rebuild System A pools that are only archived as logs.
  for name in refcoco refcocoplus refcocog; do
    hf_get "jxu124/${name}" "${WORK}/data/${name}" dataset "data/*"
  done
  ( cd "${WEIGHTS}" && find . -maxdepth 2 -name "*.json" -exec sha256sum {} + ) > "${WORK}/weights_sha256.txt"
}

inputs() {
  local R="${REPO}/experiments/cs2_audit/results/system_a_records_v2_20261016"
  "${PY}" -m experiments.cs2_audit.prepare_system_b_inputs system-a \
    --manifest "RefCOCO+=${R}/RefCOCOplus_reduced_seed20270822.csv" \
    --manifest "RefCOCOg=${R}/RefCOCOg_reduced_seed20270822.csv" \
    --manifest "RefCOCO=${R}/RefCOCO_reduced_seed20270822.csv" \
    --logs-root "${SYSA_LOGS}" --out-dir "${WORK}/inputs/g1"
  "${PY}" -m experiments.cs2_audit.prepare_system_b_inputs refclef \
    --parquet-glob "${WORK}/data/refclef/data/refclef_berkeley_test-*.parquet" \
    --image-dir "${WORK}/data/refclef_images" --out-dir "${WORK}/inputs/g3"
  "${PY}" -m experiments.cs2_audit.prepare_system_b_inputs verifier-pool \
    --train-parquet "RefCOCO=$(ls ${WORK}/data/refcoco/data/train-*.parquet)" \
    --train-parquet "RefCOCO+=$(ls ${WORK}/data/refcocoplus/data/train-*.parquet)" \
    --train-parquet "RefCOCOg=$(ls ${WORK}/data/refcocog/data/train-*.parquet)" \
    --exclude-parquet "${SYSA_LOGS}/cs2_scaleup_*/*_scaleup_*.parquet" \
    --exclude-parquet "${EXTRA_EXCLUDE}" \
    --exclude-logs "${SYSA_LOGS}/cs2_scaleup_*/scores/*_scores.json" --exclude-logs "${SYSA_LOGS}/cs2_scaleup_*/candidates/*.json" \
    --images 4000 --out-dir "${WORK}/inputs/g2"
}

protocol() {  # writes configs/system_b_protocol_v1.json; copy it back, commit and push before any label join
  local out="${REPO}/experiments/cs2_audit/configs/system_b_protocol_v1.json"
  [[ -e "${out}" ]] && { echo "${out} exists; not overwriting" >&2; exit 4; }
  "${PY}" - "${out}" "${WEIGHTS}" "${DINO}" "${WORK}/inputs" "${GPUS}" <<'PYEOF'
import datetime, glob, hashlib, json, socket, sys
from pathlib import Path
import torch, transformers
from experiments.cs2_audit.system_b import protocol_settings

out, weights, dino, inputs, gpus = sys.argv[1:]

def revision(local_dir):
    # snapshot_download(local_dir=...) leaves one metadata file per file: commit hash on the first line.
    hashes = {Path(p).read_text().splitlines()[0] for p in glob.glob(f"{local_dir}/.cache/huggingface/download/**/*.metadata", recursive=True)}
    if len(hashes) != 1:
        raise SystemExit(f"{local_dir}: expected one revision, found {sorted(hashes)}")
    return hashes.pop()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

models = {
    "vlm": ("OpenGVLab/InternVL3-8B-hf", f"{weights}/InternVL3-8B-hf"),
    "detector": ("IDEA-Research/grounding-dino-base", dino),
    "siglip": ("google/siglip-so400m-patch14-384", f"{weights}/siglip-so400m-patch14-384"),
    "clip": ("openai/clip-vit-large-patch14-336", f"{weights}/clip-vit-large-patch14-336"),
}
protocol = protocol_settings()
protocol["models"] = {k: {"repo": repo, "revision": revision(path),
                          "config_sha256": sha(f"{path}/config.json")} for k, (repo, path) in models.items()}
protocol["models"]["vlm"].update({"dtype": "bfloat16", "crop_to_patches": True, "max_patches": 6})
protocol["alpha"] = 0.05
protocol["delta"] = 0.05
protocol["selection_rule"] = ("select_policy on the selection labels only: among rules with >= 1 switch and observed "
                              "conditional harm <= alpha, maximize net accuracy gain (ties: higher coverage, then lower "
                              "threshold); if none passes, take the largest observed net gain and still certify it")
protocol["certificate"] = "one-sided Clopper-Pearson upper bound on conditional harm at level delta for the one frozen rule"
protocol["inputs"] = {}
for arm in ("g1", "g2", "g3"):
    summary = json.loads(Path(f"{inputs}/{arm}/inputs_summary.json").read_text())
    protocol["inputs"][arm] = {"records_sha256": summary["records_sha256"], "gt_sha256": sha(f"{inputs}/{arm}/gt.csv"),
                               **{k: summary[k] for k in ("counts", "split_counts", "records_by_benchmark") if k in summary}}
protocol["host"] = {"name": socket.gethostname(), "gpus": gpus, "torch": torch.__version__, "transformers": transformers.__version__}
protocol["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
protocol["threshold_families"] = {}
protocol["frozen_rules"] = {}
Path(out).write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
print("wrote", out)
PYEOF
}

records_for() {
  case "$1" in
    g1) echo "${WORK}/inputs/g1/records.csv" ;;
    g2) echo "${WORK}/inputs/g2/records.csv" ;;
    g3) echo "${WORK}/inputs/g3/records.csv" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

image_root_for() {
  case "$1" in g3) echo "/" ;; *) echo "${COCO}" ;; esac
}

generation_for() {  # the records a feature run reads: an arm's generation shards, or System A's records
  case "$1" in
    sysa) echo "${WORK}/inputs/sysa/records.jsonl" ;;
    g1|g2|g3) ls "${WORK}"/generation/"$1"_shard*.jsonl ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

use_gpu() {  # use_gpu WORKER NEED_MIB: pin the worker to its GPU; refuse another user's GPU or one without NEED_MIB free
  local worker="$1" need="$2" gpu uuid pid owner free
  local -a gpu_list
  IFS=, read -r -a gpu_list <<< "${GPUS}"
  gpu="${gpu_list[$(( worker % ${#gpu_list[@]} ))]}"
  uuid="$(nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d ' ' | awk -F, -v u="${uuid}" '$1 == u {print $2}'); do
    owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ "${owner}" != "$(id -un)" ]]; then  # an invisible process counts as another user's
      echo "GPU ${gpu} runs a process of ${owner:-another user}; not starting" >&2
      exit 3
    fi
  done
  free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free < need )) && [[ "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    echo "GPU ${gpu} has ${free} MiB free, needs ${need}; not starting" >&2
    exit 3
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  echo "worker ${worker} -> GPU ${gpu}"
}

lock_or_busy() {  # lock_or_busy FILE: hold an exclusive lock on FILE for this step, or exit 3 if another run holds it
  exec 9>"$1"
  flock -n 9 || { echo "$1 is held by another run; not starting" >&2; exit 3; }
}

pilot() {
  use_gpu "${PILOT_WORKER:-0}" 20000
  "${PY}" -m experiments.cs2_audit.system_b --records "$(records_for g1)" --image-root "${COCO}" \
    --model-path "${WEIGHTS}/InternVL3-8B-hf" --detector-path "${DINO}" \
    --output-jsonl "${WORK}/pilot/g1_train_pilot.jsonl" --splits train --max-records 60 \
    2>&1 | tee "${WORK}/logs/pilot.log"
  "${PY}" - "${WORK}/pilot/g1_train_pilot.jsonl" <<'PYEOF'
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok = [r for r in rows if "error" not in r]
print({"records": len(rows), "errors": len(rows) - len(ok),
       "default_parsed": sum(r["default_box"] is not None for r in ok),
       "sample_parsed_mean": statistics.mean(sum(b is not None for b in r["sample_boxes"]) for r in ok),
       "candidates_mean": statistics.mean(len(r["candidates"]) for r in ok),
       "passes_mean": statistics.mean(r["passes"] for r in ok),
       "seconds_mean": statistics.mean(r["seconds"] for r in ok),
       "peak_memory_mb": max(r["peak_memory_mb"] or 0 for r in ok),
       "prefix_cached": sum(r["prefix_cached"] for r in ok),
       "listwise_switches": sum((r.get("listwise_switch_score") or 0) > 0 for r in ok),
       "pairwise_switches": sum((r.get("switch_score") or 0) > 0 for r in ok)})
PYEOF
}

generate() {
  local arm="$1" worker="$2" workers="$3"
  lock_or_busy "${WORK}/generation/${arm}_shard${worker}of${workers}.lock"
  use_gpu "${worker}" 20000
  "${PY}" -m experiments.cs2_audit.system_b --records "$(records_for "${arm}")" --image-root "$(image_root_for "${arm}")" \
    --model-path "${WEIGHTS}/InternVL3-8B-hf" --detector-path "${DINO}" \
    --output-jsonl "${WORK}/generation/${arm}_shard${worker}of${workers}.jsonl" \
    --shard-index "${worker}" --num-shards "${workers}" \
    >> "${WORK}/logs/${arm}_shard${worker}of${workers}.log" 2>&1
}

sysa() {  # label-free: candidate boxes, sources and System A's own switch decision; no IoU or ground truth
  "${PY}" -m experiments.cs2_audit.verifier export-system-a --scores-root "${SYSA_LOGS}" \
    --records "$(records_for g1)" --output "${WORK}/inputs/sysa/records.jsonl"
}

features() {
  local arm="$1"
  local -a inputs
  if [[ "${arm}" == sysa && ! -s "${WORK}/inputs/sysa/records.jsonl" ]]; then sysa; fi
  lock_or_busy "${WORK}/features/${arm}_pairs.lock"
  mapfile -t inputs < <(generation_for "${arm}")
  use_gpu "${FEATURE_WORKER:-0}" 3500
  "${PY}" -m experiments.cs2_audit.verifier features \
    --generation "${inputs[@]}" --image-root "$(image_root_for "${arm}")" \
    --siglip "${WEIGHTS}/siglip-so400m-patch14-384" --clip "${WEIGHTS}/clip-vit-large-patch14-336" \
    --output "${WORK}/features/${arm}_pairs.jsonl" >> "${WORK}/logs/features_${arm}.log" 2>&1
}

wait_exit() {  # wait_exit LOG...: block until every log has an EXIT= line (poll every POLL seconds)
  local f
  for f in "$@"; do
    until grep -q "^EXIT=" "${f}" 2>/dev/null; do sleep "${POLL:-120}"; done
  done
}

wait_ok() {  # wait_ok LOG...: wait_exit, then fail unless every log's last EXIT line is EXIT=0
  local f
  wait_exit "$@"
  for f in "$@"; do
    [[ "$(grep "^EXIT=" "${f}" | tail -n 1)" == "EXIT=0" ]] || { echo "${f} did not end with EXIT=0" >&2; return 1; }
  done
}

run_logged() {  # run_logged LOG STEP ARGS...: run a step, retry while its GPU is taken, append EXIT=<code>
  local log="$1" code
  shift
  if grep -q "^EXIT=0" "${log}" 2>/dev/null; then
    echo "skipping $*: ${log} already ends a successful run"
    return 0
  fi
  while true; do
    set +e
    bash "${REPO}/experiments/cs2_audit/vgpu_g_arms.sh" "$@"
    code=$?
    set -e
    [[ "${code}" == 3 ]] || break
    echo "GPU busy at $(date -u +%FT%TZ); retrying in 10 minutes" >> "${log}"
    sleep 600
  done
  echo "EXIT=${code}" >> "${log}"
}

queue() {  # queue WORKER NWORKERS: this GPU's G1, G2 and G3 shards back to back
  local w="$1" n="$2" L="${WORK}/logs"
  wait_exit "${L}/g1_shard${w}of${n}.log"
  run_logged "${L}/g2_shard${w}of${n}.log" generate g2 "${w}" "${n}"
  run_logged "${L}/g3_shard${w}of${n}.log" generate g3 "${w}" "${n}"
}

feature_queue() {  # feature_queue NWORKERS: V1 features on GPU 2 beside the generation runs, each arm once it is complete
  local n="${1:-2}" L="${WORK}/logs" arm i
  local -a logs
  for arm in g1 sysa g2 g3; do
    logs=()
    if [[ "${arm}" != sysa ]]; then
      for (( i = 0; i < n; i++ )); do logs+=("${L}/${arm}_shard${i}of${n}.log"); done
      wait_ok "${logs[@]}" || { echo "${arm} generation failed; feature queue stopped" >&2; return 1; }
    fi
    FEATURE_WORKER=0 run_logged "${L}/features_${arm}.log" features "${arm}"
  done
}

requeue() {  # requeue WORKER NWORKERS: when this worker's G2 shard ends, replace its queue screen with the current code
  local w="$1" n="$2" L="${WORK}/logs" gpu t
  local -a gpu_list
  IFS=, read -r -a gpu_list <<< "${GPUS}"
  gpu="${gpu_list[$(( w % ${#gpu_list[@]} ))]}"
  POLL=5 wait_exit "${L}/g2_shard${w}of${n}.log"
  screen -S "garms_queue_w${w}" -X quit || true
  for (( t = 0; t < 24; t++ )); do  # up to 2 minutes for the old queue's processes to release the GPU
    (( $(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') >= 20000 )) && break
    sleep 5
  done
  echo "--- requeued $(date -u +%FT%TZ) at $(git rev-parse --short HEAD)" >> "${L}/queue_w${w}.log"
  screen -dmS "garms_queue_w${w}" bash -c "bash ${REPO}/experiments/cs2_audit/vgpu_g_arms.sh queue ${w} ${n} >> ${L}/queue_w${w}.log 2>&1"
}

mkdir -p "${WORK}/pilot" "${WORK}/generation" "${WORK}/features"
step="${1:-preflight}"; shift || true
"${step}" "$@"
