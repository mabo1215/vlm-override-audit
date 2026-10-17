# System A settings

System A is the re-ranker of the paper's original study (Table 1).  Every
setting below was recovered from its generation code and the raw logs of the
audited run.  The generation pipeline itself is not part of this release;
its outputs are released as candidate sets in `data/system_a/`.

| Field | Value |
| --- | --- |
| Default box $b(x)$ | Greedy answer (at most 128 new tokens; the checkpoint's repetition penalty of 1.05 applies) of Qwen2.5-VL-7B-Instruct to `Locate the object described by: '<expression>'. Respond with JSON: {"bbox_2d": [x1, y1, x2, y2]} in absolute pixel coordinates.`  Images are resized to multiples of 28 pixels, at most 1,003,520 pixels. |
| Box conversion | The first four numbers of the answer, divided by the original image size.  Unparseable answers (9, 17 and 7 records on RefCOCO+, RefCOCOg and RefCOCO) fall back to the top detector proposal.  The VLM answers in the frame of its resized input, so the default box is scaled by at most 5%; the paper's Appendix E shows that converting it in the resized frame changes no certification result. |
| Candidate set | The default plus GroundingDINO-base proposals for the lower-cased expression (box and text thresholds 0.25, at most 50 by score); a proposal with IoU >= 0.95 to an earlier candidate is dropped; at most 12 candidates.  No sampled answers.  Mean 2.1--2.6 and median 2 candidates per record. |
| Pairwise scoring | For each ordered pair of boxes, the VLM sees both boxes in pixel coordinates and `Which region better shows '<expression>'? Answer with only 'A' or 'B'.`  The log-odds log p(A) - log p(B) of the next token is averaged over the two orders with opposite signs, and a candidate's preference theta is its mean over all candidates (centred). |
| Alternative and score | c(x) is the candidate with the highest theta; s(x) = theta_c - theta_b in nats; a gap below 1e-6 keeps the default and is stored as s = 0. |
| Listwise score | Recorded but not used for switching: all candidates listed with letters and `Which candidate region shows '<expression>'? Answer with only the letter.`; letter probabilities renormalized and averaged over two random orders.  Audited post hoc in the paper (Section 5.6, Table A3). |
| Use of ground truth | None during generation or scoring.  IoU with the ground-truth box is computed afterwards, only to label help and harm. |
| Threshold family | tau in {0.4, 0.5, ..., 0.9}, fixed before any audited generation. |
| Cost | n(n-1) pairwise forward passes for n candidates (3.1--5.8 per record on average) and two listwise passes when n >= 2, after one generation pass; 3.3 to 4.5 hours per benchmark on one RTX 3090 with 48 GB. |
