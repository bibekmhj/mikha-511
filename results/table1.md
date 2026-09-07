# Mikha-Bench baseline — `yolov8n-seg (COCO weights, mask_area_frac score)`

Generated: 2026-09-07T16:48:24+00:00

Default threshold: **0.050**  Target recall for FAR: **0.90**


| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | FAR@Recall 90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 110 | 63 | 47 | 0.125 | 0.063 | 0.084 | 0.159 | 0.872 |
| rain | 110 | 63 | 47 | 0.206 | 0.111 | 0.144 | 0.232 | 0.915 |
| fog | 110 | 63 | 47 | 0.143 | 0.063 | 0.088 | 0.143 | 0.787 |
| night | 110 | 63 | 47 | 0.158 | 0.048 | 0.073 | 0.194 | 0.596 |
| glare | 110 | 63 | 47 | 0.125 | 0.063 | 0.084 | 0.143 | 0.872 |
| jpeg | 110 | 63 | 47 | 0.152 | 0.079 | 0.104 | 0.163 | 0.872 |

**Interpretation notes:**

* FAR (false-alert rate) is the fraction of non-flood images incorrectly alerted at the smallest threshold that reaches the target recall.
* AUROC = 0.5 means detector output is uncorrelated with the flood label.
* On COCO-pretrained weights the detector is not a water detector; low F1/AUROC here is the plan-expected baseline and triggers the fine-tune step (gate: mIoU < 0.6).
