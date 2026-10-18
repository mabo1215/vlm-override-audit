# Provenance timeline

The order in which records were generated, protocols fixed and labels read.
Section numbers refer to the submitted paper.

| Date | Event | Consequence |
| --- | --- | --- |
| 2026-07-28 | An earlier analysis by the generator project reads labels on the RefCOCO+ pool: 15 rules (pairwise, listwise, verification x 5 gates), a net-gain lower bound at error level 10%, selection and certification on the same records.  A listwise rule is selected, but the pool repeats 640 calibration and 500 confirmation records. | RefCOCO+ certification images are not strictly unseen; disclosed in Section 4.4.  Earlier exposure can only favour certification. |
| 2026-07-29 | Same analysis on the corrected pool (3,198 + 1,500 distinct records): no listwise rule passes; verification selected; confirmation net-gain bound negative. | The earlier listwise selection was an artefact of the defective pool. |
| 2026-08-22 | Four-way protocol, selection rule and threshold family {0.4, ..., 0.9} fixed for System A. | Fixed before any audited generation. |
| 2026-08-22 | RefCOCO+ regenerated (the audited records). | New generation run; scores differ from the July run. |
| 2026-08-22 | Manifest builder with the pairwise score as routing score. | The pairwise score is the pipeline's default routing score. |
| 2026-08-22 | First RefCOCO+ audit, before the one-expression reduction. | Labels read once on unreduced records. |
| 2026-08-22 | One-expression-per-image reduction (legacy source-row rule). | All reported audits use reduced records. |
| 2026-08-22/23 | RefCOCOg and RefCOCO pools drawn and generated under the full protocol. | No earlier labelled analysis on record. |
| 2026-10-16 | System A's records rebuilt with the scored-default IoU and the tie tolerance; default-box frame error quantified. | Selected rules and certification counts unchanged. |
| 2026-10-16 | Post hoc listwise check of System A on the audited records (labels already read). | Not certified on any benchmark or reduction; the pairwise score remains the audited score. |
| 2026-10-16 | System B's protocol file committed before any System B label was read; generation of G1, G2 and G3 on two RTX 3090 GPUs. | See `docs/system_b.md` and `docs/protocol_history/`. |
| 2026-10-16 to 2026-10-18 | System B's label joins in the order fixed by the protocol file (G1: training, selection, certification, confirmation; RefCLEF: transfer routes first, then the same order). | Each join is logged with the protocol's SHA-256 in `label_join_log.jsonl`. |
| 2026-10-18 | Figure 1 examples re-selected on the audited records with a licence check (same seed and order as the 2026-08-22 pre-declaration). | All three 2026-08 picks had NonCommercial licences and were skipped in the declared order. |
