# M11-b — Live corpus head-only fine-tune

**Base checkpoint:** `models/finetune_v1/best.pt`
**Out checkpoint:** `models/finetune_live/best.pt`
**Labels:** 265 (flood=124, nonflood=141) on 4 event cams
**Protocol:** freeze backbone, train final linear head only; 15 epochs, lr=0.001, batch=16, seed=42.

## Leave-one-camera-out cross-validation

| Held-out camera | n_test | AUROC | F1 | Precision | Recall | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deldot_scam013_dewey_towers` | 70 | 0.910 | 0.903 | 0.848 | 0.966 | 28 | 5 | 1 | 36 |
| `deldot_scam098_indian_river_inlet_n` | 93 | 1.000 | 0.761 | 1.000 | 0.614 | 35 | 0 | 22 | 36 |
| `deldot_scam169_event_20261126` | 65 | 0.977 | 0.945 | 1.000 | 0.897 | 26 | 0 | 3 | 36 |
| `deldot_pcam4061_event_20261126` | 37 | 0.671 | 0.400 | 0.364 | 0.444 | 4 | 7 | 5 | 21 |
| **mean** | – | **0.889** | **0.752** | – | – | – | – | – | – |

## Next step

Re-run M11 with the retrained head:

```bash
rm results/live_predictions.json   # invalidate score cache
python scripts/run_eval_real.py --checkpoint models/finetune_live/best.pt
```
