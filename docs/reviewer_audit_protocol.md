# Reviewer Audit Protocol

This repository now includes an explicit audit path for the two highest-risk
protocol questions raised during review.

## 1. Test-set anomaly-label resampling

The paper-protocol loaders construct splits in this order:

1. choose chronological raw-row intervals without reading anomaly labels;
2. generate sliding windows independently inside train, calibration, and test;
3. keep the complete chronological test stream;
4. use labels only after window construction for final metric reporting.

For SWaT, WADI, HAI, and BATADAL, every client dictionary contains a
`split_audit` record with:

- raw source file names;
- raw row start/end offsets for train, calibration, and test;
- window length and stride;
- a machine-readable `label_usage` declaration.

Run:

```bash
python audit_protocol.py --dataset wadi --seed 42
python audit_protocol.py --dataset hai --seed 42
python audit_protocol.py --dataset batadal_small --seed 42 --label-mode last
python audit_protocol.py --dataset swat --seed 42
```

The script writes:

- `<dataset>_seed<seed>_summary.json`
- `<dataset>_seed<seed>_windows.csv`
- `<dataset>_seed<seed>_anchor_identity.csv`

The window CSV lists each generated window's raw row interval. These indices are
the evidence that test windows come from a contiguous test stream rather than
from label-stratified normal/anomalous pools.

## 2. Anchor sequence identity

The audit script also compares anchor tensors across clients for train,
calibration, and test splits. The output reports whether anchor blocks are
bitwise identical after each client's train-only normalization.

Important interpretation:

- In SWaT, WADI, HAI, and BATADAL, anchors are intentionally replicated
  public-state or public-context streams from a single centralized benchmark,
  not independent factory/site sensors with only semantic alignment.
- Therefore, these experiments support the narrower claim of heterogeneous
  node subsets under shared public context.
- They should not be described as proving performance on multiple independent
  factories whose same-semantics anchors have different local values and
  asynchronous time axes.

This distinction should be stated explicitly in any revision. A stronger
multi-site claim requires additional experiments where each client owns
different but semantically aligned anchor streams.
