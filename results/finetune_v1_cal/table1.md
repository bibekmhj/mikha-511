# Mikha-Bench baseline - `water-classifier (mobilenet_v3_small, ImageNet-pretrained, fine-tuned; weights=models/finetune_v1/best.pt)`

Generated: 2026-09-08T04:17:10+00:00

Default threshold: **0.500**  Target recall for FAR: **0.90**


| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | FAR@Recall 90% | ECE (raw) | ECE (Platt) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 110 | 63 | 47 | 0.833 | 0.873 | 0.853 | 0.923 | 0.277 | 0.085 | 0.083 |
| rain | 110 | 63 | 47 | 0.685 | 0.968 | 0.803 | 0.939 | 0.149 | 0.190 | 0.137 |
| fog | 110 | 63 | 47 | 0.778 | 0.889 | 0.830 | 0.901 | 0.426 | 0.106 | 0.101 |
| night | 110 | 63 | 47 | 0.740 | 0.905 | 0.814 | 0.882 | 0.404 | 0.114 | 0.069 |
| glare | 110 | 63 | 47 | 0.824 | 0.889 | 0.855 | 0.926 | 0.277 | 0.080 | 0.128 |
| jpeg | 110 | 63 | 47 | 0.833 | 0.873 | 0.853 | 0.924 | 0.298 | 0.077 | 0.097 |

**Interpretation notes:**

* FAR (false-alert rate) is the fraction of non-flood images incorrectly alerted at the smallest threshold that reaches the target recall.
* AUROC = 0.5 means detector output is uncorrelated with the flood label.
* ECE is the equal-width-bin expected calibration error (Guo et al. 2017, 10 bins). Lower is better; 0.0 = perfectly calibrated.
* Platt scaling is fit on the val split's scores per degradation, then applied to the test split's scores. Both ECE numbers are computed on test.
