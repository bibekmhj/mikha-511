# M11 — Real-data evaluation of the persistence gate

**Classifier checkpoint:** `models/finetune_v1/best.pt`
**Gate:** PersistenceGate(window=10, k=6, clear_grace=3)
**Decision threshold (baseline / gate fixed-threshold):** 0.5
**Event window:** 20261126, 20261127 (Thanksgiving nor'easter 2026)

## Pooled across evaluation cameras

| Method | Precision | Recall | F1 | FAR @ 90% recall | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (per-frame) | 0.480 | 0.984 | 0.646 | 0.816 (thr=0.66) | 122 | 132 | 2 | 9 |
| Proposed (persistence gate) | 0.460 | 0.968 | 0.623 | 0.965 (thr=0.65) | 120 | 141 | 4 | 0 |

**Labeled rows scored:** 265 (flood=124, nonflood=141)

## Per-camera

### `deldot_scam013_dewey_towers`

flood=29, nonflood=41

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 0.414 | 1.000 | 0.586 | 1.000 (thr=0.65) |
| Gate | 0.406 | 0.966 | 0.571 | 1.000 (thr=0.65) |

### `deldot_scam098_indian_river_inlet_n`

flood=57, nonflood=36

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 0.636 | 0.982 | 0.772 | 0.667 (thr=0.66) |
| Gate | 0.609 | 0.982 | 0.752 | 0.917 (thr=0.65) |

### `deldot_scam169_event_20261126`

flood=29, nonflood=36

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 0.475 | 1.000 | 0.644 | 0.722 (thr=0.72) |
| Gate | 0.438 | 0.966 | 0.602 | 0.944 (thr=0.75) |

### `deldot_pcam4061_event_20261126`

flood=9, nonflood=28

| Method | Precision | Recall | F1 | FAR @ 90% R |
|---|---:|---:|---:|---:|
| Baseline | 0.229 | 0.889 | 0.364 | 1.000 (thr=0.41) |
| Gate | 0.222 | 0.889 | 0.356 | – |
