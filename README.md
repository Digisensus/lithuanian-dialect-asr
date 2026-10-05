# lithuanian-dialect-asr

Code, training recipes, evaluation records and analysis for the study **Lithuanian speech
recognition: effects of dialect training and transcript spelling** (study code `ltd26`).

| | |
|---|---|
| Datasets | [dialect speech (v2.0)](https://huggingface.co/datasets/Digisensus/lithuanian-dialect-speech-liepa-3-100h-punctuated) · [telephone speech (v2.0)](https://huggingface.co/datasets/Digisensus/lithuanian-phone-speech-liepa-3-429h-punctuated) · [evaluation bundle](https://huggingface.co/datasets/Digisensus/lithuanian-dialect-asr-eval) |
| Models | [six fine-tuned Parakeet-TDT checkpoints](#models) |

## Study summary

Does adding dialect speech improve Lithuanian speech recognition, and should dialect transcripts
use dialect or standard spelling?

We fine-tune `nvidia/parakeet-tdt-0.6b-v3` on 420.17 hours of LIEPA-3 telephone
speech and add up to 82.30 hours of LIEPA-3 dialect recordings, transcribed either
in dialect spelling or in standard spelling. A control condition adds the same amount of other
spontaneous speech instead. Training settings are fixed for every run. Dialect recognition is
scored against both spellings and with an *either-form* WER that accepts either reference form,
on a dialect test set of speakers never seen in training.

**Findings**

- **Dialect speech helps, and it is not just more data.** Against dialect references, WER falls
  from 41.62 % (telephone only) to 31.30 % with dialect
  recordings, averaged over two training runs; adding other spontaneous speech lowers it only to
  40.04 %.
- **Most of the gain comes early.** In the single-run comparison, the first 25.43 hours
  of dialect speech give about two thirds of the dialect-test improvement.
- **Spelling changes what is learned and what is measured.** Standard-spelling dialect transcripts
  give the lowest either-form WER among models also trained on telephone speech:
  25.97 %.
- **The benefit extends to customer-service calls.** On a private customer-service test set of
  unseen speakers, average WER falls from 44.89 % to 39.43 % (dialect spelling) and
  39.06 % (standard spelling).
- **Telephone speech changes little.** WER on the LIEPA-3 telephone test barely moves (that test
  may share speakers with training).

## Models

Run 1 of each headline condition. WER (%) on the dialect test (three reference views) and the
LIEPA-3 telephone test; the paper reports means over training runs.

| Model | Training data | Dialect ref. | Standard ref. | Either | Telephone |
|---|---|---:|---:|---:|---:|
| [parakeet-tdt-0.6b-lt-study-s](https://huggingface.co/Digisensus/parakeet-tdt-0.6b-lt-study-s) | Telephone only | 42.11 | 36.40 | 34.83 | 10.76 |
| [parakeet-tdt-0.6b-lt-study-s-d80-dial](https://huggingface.co/Digisensus/parakeet-tdt-0.6b-lt-study-s-d80-dial) | Telephone + dialect, dialect spelling | 31.28 | 36.29 | 27.60 | 10.80 |
| [parakeet-tdt-0.6b-lt-study-s-d80-std](https://huggingface.co/Digisensus/parakeet-tdt-0.6b-lt-study-s-d80-std) | Telephone + dialect, standard spelling | 37.52 | 27.06 | 26.06 | 10.65 |
| [parakeet-tdt-0.6b-lt-study-s-x80](https://huggingface.co/Digisensus/parakeet-tdt-0.6b-lt-study-s-x80) | Telephone + other spontaneous (control) | 39.90 | 33.86 | 32.18 | 10.31 |
| [parakeet-tdt-0.6b-lt-study-d80-dial](https://huggingface.co/Digisensus/parakeet-tdt-0.6b-lt-study-d80-dial) | Dialect only, dialect spelling | 27.18 | 35.32 | 24.43 | 28.20 |
| [parakeet-tdt-0.6b-lt-study-d80-std](https://huggingface.co/Digisensus/parakeet-tdt-0.6b-lt-study-d80-std) | Dialect only, standard spelling | 35.15 | 22.72 | 21.91 | 19.63 |

## Datasets

| Dataset | Content |
|---|---|
| [lithuanian-dialect-speech-liepa-3-100h-punctuated](https://huggingface.co/datasets/Digisensus/lithuanian-dialect-speech-liepa-3-100h-punctuated) | v2.0: speaker-disjoint splits, standard-spelling layer with word alignment and change tags |
| [lithuanian-phone-speech-liepa-3-429h-punctuated](https://huggingface.co/datasets/Digisensus/lithuanian-phone-speech-liepa-3-429h-punctuated) | v2.0: dev/test splits and reference layer |
| [lithuanian-dialect-asr-eval](https://huggingface.co/datasets/Digisensus/lithuanian-dialect-asr-eval) | every test set x reference layer, every model's hypotheses, scores; control-pool ids |

## Repository

| Folder | Content |
|---|---|
| `code/ltd26/` | split and recipe builders, speaker clustering, standardiser (`std_*`), scorer (`score.py`) |
| `code/tests/` | scorer and registry tests |
| `analysis/` | results, tables and figures, read only from `registry/scores.csv` |
| `recipes/` | one JSON per training condition |
| `splits/v1/` | frozen clip-id lists with sha256 manifests |
| `registry/` | datasets, splits, layers, recipes, runs, evaluations, scores |
| `protocol.md` | the pre-registered protocol, frozen before any trained model was scored |

Every number in the paper is generated from `registry/scores.csv` by `analysis/`. To rerun the
analysis, place the reference layers and hypotheses from the evaluation bundle in `layers/` and
`hyps/`. Training ran on an internal training hub whose orchestration is not included; the recipes
and each model's `config.json` give the full training spec.

The customer-service test set is private: only aggregate scores are included
(`analysis/calls_results.json`, `registry/scores.csv`).

## License

Code: MIT (see `LICENSE`). Models and datasets: see their Hugging Face cards.
