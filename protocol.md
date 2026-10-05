# ltd26 protocol: dialect data in Lithuanian ASR fine-tuning

Status: **frozen 2026-10-01 (git tag `protocol-v1`)**, after all 18 training runs finished and
before any trained model was scored on a test set (only the stock zero-shot baseline had been
scored, to verify the pipeline). Later changes are reported as deviations in the paper.

## 1. Question

When a pretrained multilingual ASR model is fine-tuned for Lithuanian, what does dialect speech in
the training data do to recognition of standard and dialect speech? How much of the measured effect
depends on transcribing dialect speech in standard word forms or in spoken dialect forms?

## 2. Hypotheses

| RQ | Hypothesis | Primary test |
|---|---|---|
| RQ1 dose | Dialect WER falls steeply with the first dialect hours, then flattens; standard WER rises slightly or not at all. | Dose curve `S`, `S+D12`, `S+D25`, `S+D50`, `S+D80` (dialect spelling); 2 seeds at the endpoints, 1 in between (trend over doses). |
| RQ2 control | On dialect tests `S+D80` beats `S+X80`; on standard tests `S+X80` ≥ `S+D80`. | Paired speaker bootstrap, `S+D80.dial` vs `S+X80`. |
| RQ3 convention | (a) The either-form gain survives with standard targets; (b) the dialect gain shrinks under standard references; (c) standard targets cost less on standard speech. | Convention matrix: target (`.dial`, `.std`) × reference view (dialect, standard, either). |
| RQ4 region | Dialect data transfers to a region left out of training; least for Žemaitija. | `S+D80-R.std` vs `S+D50.std` and `S+D80.std`, per region. |
| RQ5 measurement | Seen-speaker WER is clearly lower than unseen-speaker WER. | `dial.seen.v1` vs `dial.test.v1` for the same models. |

## 3. Data (frozen before any main run)

- Dialect splits `splits/v1` (seed 2026, sha256 in `MANIFEST.sha256`, registry `splits.csv`):
  test 12 speakers (3 per region, ~5.0 h), dev 8 speakers (~2.7 h), seen 1.0 h from 12 training
  speakers, train 97 speakers (82.3 h). The remaining 9.5 h of test/dev speakers are excluded from
  everything. Nested dose subsets D12 ⊂ D25 ⊂ D50 ⊂ D80 are sampled by speaker, stratified by
  region (`dose_order.txt`).
- Phone speech: `phone.test.v1` (4.36 h), `phone.dev.v1` (4.30 h) and `phone.train.v1` (420.2 h) follow
  the published random split of the phone dataset (`sha1(id) mod 100`). LIEPA-3 has no phone speaker
  ids and embedding clustering could not recover callers reliably,
  so these splits are **not speaker-disjoint**; train clips duplicating test/dev audio or transcripts
  are removed. Phone-test results are in-domain estimates and secondary; the claims rest on the
  speaker-disjoint dialect test and the public benchmarks. Control pool `xspon.train.v1`: 82.4 h of
  LIEPA-3 spontaneous speech from non-phone sources (radio, dictaphone, TV), matched to `dial.train.v1`
  by gender × age group × clip length, clips repeating a phone test/dev transcript excluded.
- Public benchmarks, pinned, same normaliser: FLEURS lt_lt test (`google/fleurs` @ refs/convert/
  parquet `168de341…`, 986 clips, 2.97 h; no speaker ids, so its bootstrap resamples clips) and
  Common Voice 19.0 lt test (CC0 mirror `fsicoli/common_voice_19_0` @ `590c8abe…`, 4,925 clips,
  7.15 h, 260 speakers; the official Hub repository was emptied in 2025-10). References are the
  raw transcripts (digits kept, as for every model).
- Reference layers: dialect spelling `dial.dialect.v1` (LIEPA-3 human transcription); standard
  spelling `dial.standard.v2` (semi-automatic standardisation under a written guideline, one word per
  dialect word). Phone and
  control speech use LIEPA-3's human spoken-form transcription.
- The recipe builder refuses any recipe that touches a dev/test/seen/excluded clip or a dev/test
  speaker (`registry.py`, automatic).

## 4. Conditions

Base model `nvidia/parakeet-tdt-0.6b-v3`. Identical hyperparameters for every run.

| Exp | Recipes | Seeds |
|---|---|---:|
| B | zero-shot baselines (stock model, Whisper large-v3) | – |
| E1 | `S`, `S+D80.dial` / `S+D12.dial`, `S+D25.dial`, `S+D50.dial` | 2 / 1 |
| E2 | `S+X80` | 2 |
| E3 | `S+D80.std` / `S+D25.std`, `S+D50.std` | 2 / 1 |
| E4 | `D80.dial`, `D80.std` | 1 |
| E5 | `S+D80-A.std`, `-D`, `-Z` (Suvalkija left out of E5) | 1 |

18 runs. Two seeds only where a claim rests on a small difference on the standard test
(`S`, `S+D80.dial`, `S+X80`, `S+D80.std`); training noise is estimated from these four seed
pairs (pooled) and applied to the single-seed recipes, which are points of a trend or secondary
references.

## 5. Training and selection

Full fine-tune, bf16, micro-batch 48 clips and ≤ 200 s of audio,
4 accumulation steps (192 clips per update), AdamW (0.9, 0.98), weight decay 1e-3, peak LR 1e-4,
5 % linear warm-up, cosine decay to 1e-6, SpecAugment, clips 0.5–30 s and ≤ 12 tokens/s.
**Every run trains a fixed 10,000 updates and keeps the last checkpoint**: no dev-based checkpoint
selection or early stopping, so all recipes and both spelling conventions are treated identically
and no selection bias enters. Dev (`phone.dev.v1` + `dial.dev.v1`, 2,000 clips every 1,000 updates,
full set at the end) is logged for monitoring only. Seeds 1–3 change data order, augmentation and
initialisation of new layers; the data recipe is fixed. Training text is the recipe's layer:
spoken form for phone and X, dialect or standard spelling for the dialect part.

## 6. Metrics and analysis (`code/ltd26/score.py`, versioned)

- Normaliser: lowercase, NFC, punctuation → space, whitespace collapsed.
- WER and CER pooled per test set; per-speaker mean and per-region WER reported alongside.
- Dialect test: WER against dialect spelling, against standard spelling, and either-form WER
  (a word is correct if it matches either form of its aligned reference word).
- Errors by change tag (same, kept, nasal, ending, pron).
- Uncertainty: mean ± sd over seeds (four seed pairs, pooled); 95 % CIs from a speaker-level
  bootstrap (10,000 resamples) of paired differences on the same test clips. Speakers: the dialect
  speaker code; Common Voice client ids; phone test and FLEURS have none (resampling by clip).
- Decoding: every model decodes every test set once (greedy, the model's own
  settings); hypotheses are stored under `hyps/` and scored only by `score.py` ≥ 1.1. Scorer check
  2026-10-01: `score.py` and the training framework's scorer give 42.929 on the same 4,206 clips / 37,206 words.
- Zero-shot baselines B: stock `nvidia/parakeet-tdt-0.6b-v3` and `openai/whisper-large-v3`.
- Primary comparisons (RQ1–RQ3) are the paired differences named in §2. Everything else is
  secondary and reported as such.
- Seed rule: if seed sd ≳ one third of an effect of interest, seeds are added for the recipes of that
  comparison (a second seed of a single-seed recipe first, then a third seed of the key ones).

## 7. Reporting rules

Every run is logged (`registry/runs.csv`), including failed and repeated runs; exclusions are
reported. Every number in the paper is regenerated from `registry/scores.csv` on the final commit.
Hypotheses are evaluated on test only after all runs of the experiment are finished.

## 8. Known limitations (stated in the paper)

In-house annotators (no professional dialectologist); one base-model family; phone splits not
speaker-disjoint (no speaker ids; clustering attempt reported); the standard layer is semi-automatic.
