import csv, json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ltd26"))
import score

LAYER = {"dial.test.v1": "layers/dial.standard.v2.jsonl", "dial.seen.v1": "layers/dial.standard.v2.jsonl",
         "phone.test.v1": "layers/phone.spoken.v1.jsonl", "fleurs.lt.test.v1": "layers/fleurs.lt.test.v1.jsonl",
         "cv.lt.test.v1": "layers/cv.lt.test.v1.jsonl"}
STD_TESTS = ["phone.test.v1", "fleurs.lt.test.v1", "cv.lt.test.v1"]
KEY = {"dialect": "e_dial", "standard": "e_std", "either": "e_either"}
NBOOT = 10000
SEED_PAIRS = ["S", "S+D80.dial", "S+X80", "S+D80.std"]

_cache: dict = {}


def per_clip(model, split):
    k = (model, split)
    if k not in _cache:
        ev = EVALS[k]
        _res, per, ids = score.score_file(str(ROOT / ev["hyps_file"]), split, str(ROOT / LAYER[split]))
        _cache[k] = (per, ids)
    return _cache[k]


def wer(model, split, view="dialect", region=None):
    per, ids = per_clip(model, split)
    if region:
        ids = [i for i in ids if i[0] == region]
    e = sum(per[i][KEY[view]] for i in ids)
    n = sum(per[i]["n_std" if view == "standard" else "n"] for i in ids)
    return 100 * e / n if n else None


def paired(a, b, split, view="dialect", region=None):
    pa, ids = per_clip(a, split)
    pb, _ = per_clip(b, split)
    if region:
        ids = [i for i in ids if i[0] == region]
    lo, hi = score.bootstrap(pa, ids, KEY[view], other=pb, n_boot=NBOOT)
    d = round(wer(a, split, view, region) - wer(b, split, view, region), 2)
    return {"diff": d, "ci95": [round(lo, 2), round(hi, 2)], "a": a, "b": b, "test": split, "view": view,
            **({"region": region} if region else {})}


def label(run_id):
    return run_id.replace("ltd26/", "").rsplit("/a", 1)[0]


EVALS = {}
for e in csv.DictReader(open(ROOT / "registry" / "evals.csv")):
    EVALS[(label(e["run_id"]), e["split_id"])] = e
MODELS = sorted({m for m, _ in EVALS}, key=lambda m: (not m.startswith("B/"), m))


def main():
    out = {"scorer": score.VERSION, "n_boot": NBOOT}
    table = {}
    for m in MODELS:
        table[m] = {
            "dial.test": {v: wer(m, "dial.test.v1", v) for v in ("dialect", "standard", "either")},
            "dial.seen": {v: wer(m, "dial.seen.v1", v) for v in ("dialect", "either")},
            **{t.split(".v1")[0]: wer(m, t) for t in STD_TESTS}}
    out["table"] = table
    noise = {}
    for t, view in [("dial.test.v1", "dialect"), ("dial.test.v1", "either")] + [(t, "dialect") for t in STD_TESTS]:
        diffs = [wer(f"{r}/s1", t, view) - wer(f"{r}/s2", t, view) for r in SEED_PAIRS]
        sd = math.sqrt(sum(d * d for d in diffs) / (2 * len(diffs)))
        noise[f"{t.split('.v1')[0]}:{view}"] = {"diffs_s1_minus_s2": [round(d, 2) for d in diffs],
                                                 "pooled_seed_sd": round(sd, 3)}
    out["seed_noise"] = noise
    dose = ["S/s1", "S+D12.dial/s1", "S+D25.dial/s1", "S+D50.dial/s1", "S+D80.dial/s1"]
    hours = [0, 12.83, 25.43, 50.47, 82.3]
    out["RQ1_dose"] = {
        "hours": hours,
        "dial.test_dialect": [wer(m, "dial.test.v1", "dialect") for m in dose],
        "dial.test_either": [wer(m, "dial.test.v1", "either") for m in dose],
        **{f"{t.split('.v1')[0]}": [wer(m, t) for m in dose] for t in STD_TESTS},
        "endpoint_S+D80.dial_minus_S": [paired(f"S+D80.dial/s{k}", f"S/s{k}", t, v)
                                        for k in (1, 2) for t, v in [("dial.test.v1", "dialect")] + [(t, "dialect") for t in STD_TESTS]]}
    out["RQ2_control"] = [paired(f"S+D80.dial/s{k}", f"S+X80/s{k}", t, v) for k in (1, 2)
                          for t, v in [("dial.test.v1", "dialect"), ("dial.test.v1", "either")] + [(t, "dialect") for t in STD_TESTS]]
    out["RQ2_X80_minus_S"] = [paired(f"S+X80/s{k}", f"S/s{k}", t, v) for k in (1, 2)
                              for t, v in [("dial.test.v1", "dialect")] + [(t, "dialect") for t in STD_TESTS]]
    out["RQ3"] = {
        "a_std_targets_gain_either": [paired(f"S+D80.std/s{k}", f"S/s{k}", "dial.test.v1", "either") for k in (1, 2)],
        "b_dial_model_by_view": {f"S+D80.dial/s{k}": {v: wer(f"S+D80.dial/s{k}", "dial.test.v1", v)
                                                     for v in ("dialect", "standard", "either")} for k in (1, 2)},
        "b_gain_over_S_by_view": [paired(f"S+D80.dial/s{k}", f"S/s{k}", "dial.test.v1", v)
                                  for k in (1, 2) for v in ("dialect", "standard", "either")],
        "c_std_minus_dial_on_standard_tests": [paired(f"S+D80.std/s{k}", f"S+D80.dial/s{k}", t) for k in (1, 2)
                                               for t in STD_TESTS],
        "std_vs_dial_either_dialtest": [paired(f"S+D80.std/s{k}", f"S+D80.dial/s{k}", "dial.test.v1", "either")
                                        for k in (1, 2)],
        "std_dose_either": {m: wer(m, "dial.test.v1", "either") for m in
                            ["S/s1", "S+D25.std/s1", "S+D50.std/s1", "S+D80.std/s1"]}}
    rq4 = {}
    for r in "ADSZ":
        rq4[r] = {"left_out": wer(f"S+D80-{r}.std/s1", "dial.test.v1", "either", r),
                  "all_regions_D80": wer("S+D80.std/s1", "dial.test.v1", "either", r),
                  "all_regions_D50": wer("S+D50.std/s1", "dial.test.v1", "either", r),
                  "no_dialect_S": wer("S/s1", "dial.test.v1", "either", r),
                  "vs_D80": paired(f"S+D80-{r}.std/s1", "S+D80.std/s1", "dial.test.v1", "either", r),
                  "vs_S": paired(f"S+D80-{r}.std/s1", "S/s1", "dial.test.v1", "either", r)}
    out["RQ4_region"] = rq4
    out["RQ5_seen_minus_unseen_either"] = {
        m: round(wer(m, "dial.seen.v1", "either") - wer(m, "dial.test.v1", "either"), 2)
        for m in MODELS if not m.startswith("B/")}
    json.dump(out, open(ROOT / "analysis" / "results.json", "w"), indent=1, ensure_ascii=False)
    write_md(out)
    print("wrote analysis/results.json and analysis/results.md")


def fmt_p(p):
    return f"{p['diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"


def write_md(o):
    L = [f"# ltd26 first results (generated by analysis/results.py; {o['scorer']}, {o['n_boot']} bootstrap resamples)\n",
         "WER in %. Paired differences: A − B with 95 % bootstrap CI (negative = A better). "
         "Resampling uses speakers where identified and recordings otherwise.\n",
         "## All models × test sets\n",
         "| model | dial.test dialect | standard | either | dial.seen either | phone.test | FLEURS | CV19 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, r in o["table"].items():
        d = r["dial.test"]
        L.append(f"| {m} | {d['dialect']:.2f} | {d['standard']:.2f} | {d['either']:.2f} | {r['dial.seen']['either']:.2f} | "
                 f"{r['phone.test']:.2f} | {r['fleurs.lt.test']:.2f} | {r['cv.lt.test']:.2f} |")
    L += ["\n## Seed noise (four seed pairs)\n", "| test:view | s1 − s2 (S, D80.dial, X80, D80.std) | pooled sd |", "|---|---|---:|"]
    for k, v in o["seed_noise"].items():
        L.append(f"| {k} | {', '.join(f'{d:+.2f}' for d in v['diffs_s1_minus_s2'])} | {v['pooled_seed_sd']} |")
    q = o["RQ1_dose"]
    L += ["\n## RQ1 dose (dialect spelling, seed 1)\n", "| hours | " + " | ".join(str(h) for h in q["hours"]) + " |",
          "|---|" + "---:|" * len(q["hours"])]
    for k in ("dial.test_dialect", "dial.test_either", "phone.test", "fleurs.lt.test", "cv.lt.test"):
        L.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in q[k]) + " |")
    L += ["\nEndpoint S+D80.dial − S:\n"] + [f"- s{1 if i < 4 else 2} {p['test']} ({p['view']}): {fmt_p(p)}"
                                           for i, p in enumerate(q["endpoint_S+D80.dial_minus_S"])]
    L += ["\n## RQ2 control: S+D80.dial − S+X80\n"] + [f"- {p['a'].split('/')[1]} {p['test']} ({p['view']}): {fmt_p(p)}"
                                                       for p in o["RQ2_control"]]
    L += ["\nS+X80 − S (more ordinary speech alone):\n"] + [f"- {p['a'].split('/')[1]} {p['test']}: {fmt_p(p)}"
                                                          for p in o["RQ2_X80_minus_S"]]
    r3 = o["RQ3"]
    L += ["\n## RQ3 spelling convention\n", "(a) S+D80.std − S on dial.test, either-form:"]
    L += [f"- {p['a'].split('/')[1]}: {fmt_p(p)}" for p in r3["a_std_targets_gain_either"]]
    L += ["\n(b) S+D80.dial − S on dial.test by reference view:"]
    L += [f"- {p['a'].split('/')[1]} {p['view']}: {fmt_p(p)}" for p in r3["b_gain_over_S_by_view"]]
    L += ["\n(c) S+D80.std − S+D80.dial on standard tests:"]
    L += [f"- {p['a'].split('/')[1]} {p['test']}: {fmt_p(p)}" for p in r3["c_std_minus_dial_on_standard_tests"]]
    L += ["\nS+D80.std − S+D80.dial on dial.test, either-form:"]
    L += [f"- {p['a'].split('/')[1]}: {fmt_p(p)}" for p in r3["std_vs_dial_either_dialtest"]]
    L += ["\nStandard-spelling dose (dial.test either-form): " + ", ".join(f"{m} {v:.2f}" for m, v in r3["std_dose_either"].items())]
    L += ["\n## RQ4 region transfer (dial.test either-form, test speakers of the region)\n",
          "| region | left out (S+D80-R.std) | all regions D80.std | all regions D50.std | no dialect S | left-out − D80 | left-out − S |",
          "|---|---:|---:|---:|---:|---|---|"]
    for r, v in o["RQ4_region"].items():
        L.append(f"| {r} | {v['left_out']:.2f} | {v['all_regions_D80']:.2f} | {v['all_regions_D50']:.2f} | {v['no_dialect_S']:.2f} | "
                 f"{fmt_p(v['vs_D80'])} | {fmt_p(v['vs_S'])} |")
    L += ["\n## RQ5 seen − unseen speakers (either-form WER, dial.seen − dial.test)\n"]
    L += [f"- {m}: {d:+.2f}" for m, d in o["RQ5_seen_minus_unseen_either"].items()]
    (ROOT / "analysis" / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
