# Mikha-Bench baseline — `water-classifier (mobilenet_v3_small, ImageNet-pretrained, fine-tuned; weights=models/finetune_v1/best.pt)`

Generated: 2026-09-07T21:15:07+00:00

Default threshold: **0.500**  Target recall for FAR: **0.90**


| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | FAR@Recall 90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 110 | 63 | 47 | 0.833 | 0.873 | 0.853 | 0.923 | 0.277 |
| rain | 110 | 63 | 47 | 0.685 | 0.968 | 0.803 | 0.939 | 0.149 |
| fog | 110 | 63 | 47 | 0.778 | 0.889 | 0.830 | 0.901 | 0.426 |
| night | 110 | 63 | 47 | 0.740 | 0.905 | 0.814 | 0.882 | 0.404 |
| glare | 110 | 63 | 47 | 0.824 | 0.889 | 0.855 | 0.926 | 0.277 |
| jpeg | 110 | 63 | 47 | 0.833 | 0.873 | 0.853 | 0.924 | 0.298 |

**Interpretation notes:**

* FAR (false-alert rate) is the fraction of non-flood images incorrectly alerted at the smallest threshold that reaches the target recall.
* AUROC = 0.5 means detector output is uncorrelated with the flood label.
* On COCO-pretrained weights the detector is not a water detector; low F1/AUROC here is the plan-expected baseline and triggers the fine-tune step (gate: mIoU < 0.6).
