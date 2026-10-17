# System B: preregistration and evidence of order

System B (InternVL3-8B, Section 4.2 and Table 2 of the paper) was specified
in full before it was run.  Its protocol file,
`experiments/cs2_audit/configs/system_b_protocol_v1.json`, records the model
and detector revisions, every prompt with its SHA-256, the decoding and
candidate settings, the score, alpha = delta = 0.05, the selection rule, the
input record hashes and the declared transfer test.  The file was extended
step by step, and every version was committed before the labels it governs
were read:

1. protocol committed, before any ground truth was joined;
2. threshold families (label-free percentiles of the training switch
   scores) committed;
3. training labels joined;
4. the verifier V1, the SHA-256 of its predictions and the V1-gated family
   committed (`require_verifier_gate` refuses any selection join before);
5. selection labels joined; the pairwise, listwise and V1-gated rules
   selected and frozen, and committed;
6. certification labels joined, then confirmation labels;
7. for RefCLEF, the transfer routes of G1's frozen RefCOCO rules were fixed
   and committed before any RefCLEF label was read; then steps 2--6.

`docs/protocol_history/` holds every committed version of the protocol file
with its commit time and SHA-256 (`index.json`).  Every label join in
`experiments/cs2_audit/results/system_b_20261016/label_join_log.jsonl`
records the SHA-256 of the protocol file at the time of the join, the input
and output hashes and the UTC time, so the order above can be checked:

```python
import json
versions = {r["sha256"]: r["commit"] for r in json.load(open("docs/protocol_history/index.json"))}
for line in open("experiments/cs2_audit/results/system_b_20261016/label_join_log.jsonl"):
    e = json.loads(line)
    print(e["arm"], e["benchmark"], e["stage"], e["utc"], versions.get(e["protocol_sha256"]))
```

`join_system_b_labels.py` enforces the order (it refuses a join whose
prerequisites are not committed, and joins each set once); its tests are in
`test_join_system_b_labels.py`.

## Arms

* **G1**: System B on exactly System A's COCO records (13,302 images).
* **G2**: 12,483 expressions on 4,000 further COCO training images, disjoint
  from every image of System A's pools and pilots; used only to train V1.
* **G3**: RefCLEF test split (9,999 SAIAPR TC-12 images), split 10/30/30/30;
  G3a audits System B there, G3b is the transfer test.

## Regenerating System B

`experiments/cs2_audit/vgpu_g_arms.sh` is the driver used for the run (two
RTX 3090 GPUs with 24 GB; about 84 GPU-hours).  Its host paths are defaults
that can be overridden with environment variables (`BASE`, `REPO`, `WORK`,
`VENV`, `WEIGHTS`, `COCO`, `GPUS`).  The steps are `setup`, `download`,
`inputs`, `generate ARM WORKER NWORKERS`, `features ARM` and the label joins of
`join_system_b_labels.py`; `requirements-gpu.txt` lists the GPU packages.
Generation is seeded per expression, so records do not depend on sharding.
The `inputs` step builds the record tables from System A's archived pools,
which are not released; the released records (`data/system_b/`) list every
record key, image and set, from which the input tables can be rebuilt with
the public annotations.
