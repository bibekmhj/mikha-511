# Mikha-Bench baseline — `water-classifier (mobilenet_v3_small, ImageNet-pretrained, fine-tuned; weights=models/finetune_v1/best.pt)`

Generated: 2026-09-07T21:07:27+00:00

Default threshold: **0.050**  Target recall for FAR: **0.90**


| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | FAR@Recall 90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 110 | 63 | 47 | 0.602 | 0.984 | 0.747 | 0.923 | 0.277 |
| rain | 110 | 63 | 47 | 0.590 | 0.984 | 0.738 | 0.939 | 0.149 |
| fog | 110 | 63 | 47 | 0.589 | 1.000 | 0.741 | 0.901 | 0.426 |
| night | 110 | 63 | 47 | 0.583 | 1.000 | 0.737 | 0.882 | 0.404 |
| glare | 110 | 63 | 47 | 0.596 | 0.984 | 0.743 | 0.926 | 0.277 |
| jpeg | 110 | 63 | 47 | 0.602 | 0.984 | 0.747 | 0.924 | 0.298 |

**Interpretation notes:**

* FAR (false-alert rate) is the fraction of non-flood images incorrectly alerted at the smallest threshold that reaches the target recall.
* AUROC = 0.5 means detector output is uncorrelated with the flood label.
* On COCO-pretrained weights the detector is not a water detector; low F1/AUROC here is the plan-expected baseline and triggers the fine-tune step (gate: mIoU < 0.6).
