# FedHGF final data split notes

This file records the final data/client split used by FedHGF.

## Summary

| Dataset | Dataset key | Clients | Anchor | Client dimensions | Notes |
|---|---:|---:|---|---|---|
| SWaT | `swat` | 6 | FIT101-FIT601 | 10 / 16 / 14 / 14 / 18 / 9 | Public-context anchor, all stage FITs are SCADA-visible. |
| WADI | `wadi` | 2 | all 15 zone3 variables | 37 / 108 | Physical downstream zone3 anchor; includes 3 global sensors in each client aux. |
| HAI | `hai` | 3 | P3 process, 7 variables | 45 / 29 / 19 | P1, P2, and P4 are clients; P3 is the shared water-tank anchor. |
| BATADAL | `batadal_small` | 2 | L_T1-L_T7 | 19 / 31 | Same raw data as BATADAL; `small` is a model/window variant, not a different dataset. |

## SWaT

Raw files:

- `Data/SWAT/normal.csv`
- `Data/SWAT/attack.csv`

Protocol:

- `window_len=16`
- `stride=4`
- `cal_normal_frac=0.15`
- Calibration is the final 15% of `normal.csv`, with no anomaly-label sampling.
- Test uses the full chronological `attack.csv` stream and preserves its natural anomaly rate.

Clients:

- `stage1`: anchor `FIT101-FIT601` + aux `LIT101, MV101, P101, P102`
- `stage2`: anchor `FIT101-FIT601` + aux `AIT201, AIT202, AIT203, MV201, P201-P206`
- `stage3`: anchor `FIT101-FIT601` + aux `DPIT301, LIT301, MV301-MV304, P301, P302`
- `stage4`: anchor `FIT101-FIT601` + aux `AIT401, AIT402, LIT401, P401-P404, UV401`
- `stage5`: anchor `FIT101-FIT601` + aux `AIT501-AIT504, FIT502-FIT504, P501, P502, PIT501-PIT503`
- `stage6`: anchor `FIT101-FIT601` + aux `P601, P602, P603`

Rationale:

The six FIT variables are treated as public SCADA context. Each stage owns its local FIT physically, while the full FIT set is assumed visible through the centralized SCADA broadcast.

## WADI

Raw files:

- `Data/WADI/WADI_14days_new.csv`
- `Data/WADI/WADI_attackdataLABLE.csv`

Protocol:

- `window_len=16`
- `stride=4`
- `cal_normal_frac=0.15`
- `max_train_rows=120000`
- `anchor_mode=all15`
- `client_zones=["zone1", "zone2"]`
- Calibration is the final 15% of the normal stream, with no anomaly-label sampling.
- Test uses the full chronological attack stream and preserves its natural anomaly rate.

Clients:

- `zone1`: zone1 sensors + all 15 zone3 anchors + global aux sensors
- `zone2`: zone2 sensors, including zone2A/zone2B branches + all 15 zone3 anchors + global aux sensors

Anchor:

All 15 zone3 variables are used as physical/public-state anchors. This avoids the possible criticism that anchor variables were selected by a statistical heuristic.

## HAI

Raw files:

- `Data/HAI 21.03/train*.csv.gz`
- `Data/HAI 21.03/test*.csv.gz`

Protocol:

- `processes=["P1", "P2", "P4"]`
- `anchor_process="P3"`
- `window_len=16`
- `stride=4`
- `cal_normal_frac=0.08`
- `max_train_rows=200000`
- `min_cal_anom=100`
- Calibration is the final 8% of the normal stream, with no anomaly-label sampling.
- Test uses the full chronological HAI test stream and preserves its natural anomaly rate.

Clients:

- `P1`: P3 anchor 7 + P1 aux 38 = 45 dimensions
- `P2`: P3 anchor 7 + P2 aux 22 = 29 dimensions
- `P4`: P3 anchor 7 + P4 aux 12 = 19 dimensions

Anchor:

P3 water-tank variables:

- `P3_FIT01`
- `P3_LCP01D`
- `P3_LCV01D`
- `P3_LH`
- `P3_LIT01`
- `P3_LL`
- `P3_PIT01`

## BATADAL

Raw files:

- `Data/BATADAL/BATADAL_dataset03.csv`
- `Data/BATADAL/BATADAL_dataset04.csv`

Final dataset key:

- `batadal_small`

Important:

`batadal_small` is not a different raw dataset. It uses the same BATADAL source files as `batadal`, with the paper window protocol:

- `window_len=32`
- `stride=1`
- Calibration is the final 15% of `BATADAL_dataset03.csv`, with no anomaly-label sampling.
- Test uses the full chronological `BATADAL_dataset04.csv` stream and preserves its natural anomaly rate.

Clients:

- `zoneA`: L_T1-L_T7 anchor + `F_PU1/S_PU1` through `F_PU6/S_PU6` = 19 dimensions
- `zoneB`: L_T1-L_T7 anchor + `F_PU7/S_PU7` through `F_PU11/S_PU11`, `F_V2/S_V2`, and pressure sensors `P_J*` = 31 dimensions

Anchor:

Seven tank-level variables:

- `L_T1`
- `L_T2`
- `L_T3`
- `L_T4`
- `L_T5`
- `L_T6`
- `L_T7`

## Legacy locked single-seed FedHGF results

The values below were produced by the previous tuned protocol and are retained
only for traceability. They are not expected to match the paper-protocol
implementation after the label-free chronological split, DP defaults, HAI
three-client split, local-center scoring, and signed graph residual changes.

These are the final table-level targets used while organizing the public release.

| Dataset | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|
| SWaT | 91.47 | 72.52 | 0.8090 | 0.8890 |
| WADI | 88.30 | 84.03 | 0.8611 | 0.9763 |
| HAI | 76.96 | 61.01 | 0.6806 | 0.8411 |
| BATADAL | 74.68 | 80.82 | 0.7763 | 0.8959 |

The exact value can vary slightly if CUDA/PyTorch kernels or random seeds differ, but the final script locks `--seeds 42`.
