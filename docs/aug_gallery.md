# Mikha-Aug — before / after gallery

Reproduce these samples with:

```bash
python scripts/make_aug_gallery.py --seed 1234
```

The source image is the ultralytics-bundled `bus.jpg` (a street scene). The
five canonical Mikha-Bench degradations are rendered at four severities each
(0.10, 0.35, 0.65, 0.95) with a fixed seed of 1234 so this page is
reproducible byte-for-byte.

## Source

![source](aug_samples/source.png)

## Rain-on-lens

Localised elongated droplets with a slight bright highlight; non-droplet
regions stay crisp. Occlusion budget grows with severity.

| 0.10 | 0.35 | 0.65 | 0.95 |
|---|---|---|---|
| ![](aug_samples/rain_s10.png) | ![](aug_samples/rain_s35.png) | ![](aug_samples/rain_s65.png) | ![](aug_samples/rain_s95.png) |

## Fog / spray

Depth-independent white veil, mild blur, contrast pulled toward the local
mean.

| 0.10 | 0.35 | 0.65 | 0.95 |
|---|---|---|---|
| ![](aug_samples/fog_s10.png) | ![](aug_samples/fog_s35.png) | ![](aug_samples/fog_s65.png) | ![](aug_samples/fog_s95.png) |

## Night / low-light

Gamma darkening, cool cast, additive noise floor.

| 0.10 | 0.35 | 0.65 | 0.95 |
|---|---|---|---|
| ![](aug_samples/night_s10.png) | ![](aug_samples/night_s35.png) | ![](aug_samples/night_s65.png) | ![](aug_samples/night_s95.png) |

## Glare / lens flare

Additive Gaussian bright blob biased to the upper half of the frame plus a
local saturation clip.

| 0.10 | 0.35 | 0.65 | 0.95 |
|---|---|---|---|
| ![](aug_samples/glare_s10.png) | ![](aug_samples/glare_s35.png) | ![](aug_samples/glare_s65.png) | ![](aug_samples/glare_s95.png) |

## Low-bitrate JPEG

Encode → decode at JPEG quality 35 → 15 as severity climbs; block artefacts
and colour banding appear.

| 0.10 | 0.35 | 0.65 | 0.95 |
|---|---|---|---|
| ![](aug_samples/jpeg_s10.png) | ![](aug_samples/jpeg_s35.png) | ![](aug_samples/jpeg_s65.png) | ![](aug_samples/jpeg_s95.png) |

---

**Not shown here:** the extra `ir_night` transform is exposed for exploration
but is not part of the v0.1.0 Mikha-Bench 5-class stratification.
