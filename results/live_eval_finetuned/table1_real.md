# M11 — Real-data evaluation of the persistence gate

**Classifier checkpoint:** `models/finetune_live/best.pt`
**Gate:** PersistenceGate(window=10, k=6, clear_grace=3)
**Decision threshold (baseline / gate fixed-threshold):** 0.5
**Event window:** 20261126, 20261127 (Thanksgiving nor'easter 2026)

## Pooled across evaluation cameras

| Method | Precision | Recall | F1 | FAR @ 90% recall | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (per-frame) | 0.983 | 0.935 | 0.959 | 0.000 (thr=0.82) | 116 | 2 | 8 | 139 |
| Proposed (persistence gate) | 0.982 | 0.895 | 0.937 | 0.028 (thr=0.40) | 111 | 2 | 13 | 139 |

**Labeled rows scored:** 265 (flood=124, nonflood=141)

## Per-camera

### `deldot_scam013_dewey_towers`

flood=29, nonflood=41

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 0.935 | 1.000 | 0.967 | 0.000 (thr=0.88) |
| Gate | 0.933 | 0.966 | 0.949 | 0.000 (thr=0.85) |

### `deldot_scam098_indian_river_inlet_n`

flood=57, nonflood=36

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 1.000 | 1.000 | 1.000 | 0.000 (thr=0.89) |
| Gate | 1.000 | 0.965 | 0.982 | 0.000 (thr=0.85) |

### `deldot_scam169_event_20261126`

flood=29, nonflood=36

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 1.000 | 0.931 | 0.964 | 0.000 (thr=0.60) |
| Gate | 1.000 | 0.897 | 0.945 | 0.028 (thr=0.25) |

### `deldot_pcam4061_event_20261126`

flood=9, nonflood=28

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 1.000 | 0.333 | 0.500 | 0.321 (thr=0.13) |
| Gate | 1.000 | 0.222 | 0.364 | – |
