# Live 511 held-out corpus (M10)

Plan section 4 names FL511 / TX511 live snapshots as the held-out
validation source: "Publicly viewable; ToS restrict redistribution.
Store hashes + URLs; ship labels only; run eval locally." M10 is the
data + tooling for that. The persistence-gate ablation that consumes
this data is M11.

M10 ships:

- `scripts/collect_live_511.py` - snapshot poller. Config driven, dry
  run, one-shot, and continuous modes.
- `scripts/label_live.py` - keyboard-driven hand-labeler that writes to
  `bench/live_labels.json` via atomic replace.
- `src/mikha/bench/live.py` - schema (`LiveManifestRow`), loader,
  label store, `iter_series` for per-camera time-ordered frames, and
  a small `corpus_stats` helper.
- `configs/live_cameras.yaml` - empty camera list (populate locally
  after ToS review).
- `data/live_511/README.md` - ToS notes and the labeling protocol.

M10 does NOT ship:

- Any live camera URLs. The seed config is empty on purpose. Every
  operator's ToS is per-site and you should confirm the site's rules
  before adding cameras.
- Any collected images. `data/live_511/*.jpg` and friends are
  gitignored.
- Wiring for the persistence gate, calibration, or eval on this
  corpus. That is M11 scope and is deliberately deferred.

## Corpus separation

The v0.1.0 base set in `data/manifest.csv` and this live corpus are
independent. Different schemas, different loaders, different licensing
posture:

|                         | Base set (v0.1.0)              | Live corpus (M10)                |
|-------------------------|--------------------------------|----------------------------------|
| Location                | `data/manifest.csv`            | `data/live_511/manifest.csv`     |
| Schema                  | `ManifestRow`                  | `LiveManifestRow`                |
| Images                  | Redistributable per row's license | NOT redistributed             |
| Ships in repo           | Manifest + labels + splits     | Manifest schema + labels only    |
| Time series             | No                             | Yes (`series_id`, `captured_at`) |
| Loader                  | `mikha.bench.load_manifest`    | `mikha.bench.live.load_live_manifest` |

Nothing in the base-set pipeline (training, calibration, eval, splits)
is touched by M10. That was a scope guardrail.

## Timeline expectation

Realistic single-developer pace on a FL/TX 511 collection: point the
collector at 3-5 flood-prone cameras during a storm week, poll every
30 seconds, end up with a few thousand frames. Label 100-200 of them
by hand across a couple of sessions. That is enough for M11 to have
per-camera series of ~120 frames (one hour at 30s cadence) to run
the persistence gate against.

Data collection is the bottleneck for the milestone, not code.

## Next step (M11, not yet started)

M11 will add:

- `--held-out live` mode to `scripts/run_eval.py` that consumes
  `data/live_511/manifest.csv` + `bench/live_labels.json`.
- A `PersistenceGate` wrapper in the eval driver that applies the
  K-of-N rule per series.
- Baseline vs proposed columns in `results/table1.md` per plan
  section 8.

Nothing in M10 pre-commits to a specific gate configuration; the
`LiveSeries` returned by `iter_series` is just a time-ordered tuple
of frames that any downstream code can consume.
