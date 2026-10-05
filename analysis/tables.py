import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
R = json.load(open(ROOT / "analysis" / "results.json"))
try:
    from .publication_data import table as publication_table, scores as publication_scores, customer_results
except ImportError:
    from publication_data import table as publication_table, scores as publication_scores, customer_results
T = publication_table()
STATS = json.load(open(ROOT / "splits" / "v1" / "stats.json"))
PHONE = json.load(open(ROOT / "splits" / "v1" / "phone_stats.json"))
XSPON = json.load(open(ROOT / "splits" / "v1" / "xspon_stats.json"))
SCORES = list(csv.DictReader(open(ROOT / "registry" / "scores.csv")))
RUNS = list(csv.DictReader(open(ROOT / "registry" / "runs.csv")))
TESTS = ["dial.test", "dial.seen", "phone.test", "fleurs.lt.test", "cv.lt.test"]
RECIPES = ["S", "S+D12.dial", "S+D25.dial", "S+D50.dial", "S+D80.dial", "S+X80", "S+D25.std", "S+D50.std",
           "S+D80.std", "D80.dial", "D80.std", "S+D80-A.std", "S+D80-D.std", "S+D80-S.std", "S+D80-Z.std"]
BASE = ["B/parakeet-tdt-0.6b-v3", "B/whisper-large-v3"]
macros = {}


def tex(s):
    return s.replace("+", "+").replace("_", "\\_").replace("%", "\\%")


def f2(x):
    return f"{x:.2f}"


def signed(x, nd=2):
    x = 0.0 if abs(x) < 0.5 * 10 ** -nd else x
    return f"{x:+.{nd}f}".replace("-", "\\textminus{}")


def ci(p, unit=False):
    lo, hi = p["ci95"]
    pp = "\\,pp" if unit else ""
    return f"{signed(p['diff'], 2)}{pp} [{signed(lo, 2)}, {signed(hi, 2)}]"


def wer(model, test, view="dialect"):
    v = T[model][test]
    return v[view] if isinstance(v, dict) else v


def seeds(recipe, test, view="dialect"):
    return [wer(f"{recipe}/s{k}", test, view) for k in (1, 2) if f"{recipe}/s{k}" in T]


def mean(xs):
    return sum(xs) / len(xs)


def _walk(o):
    if isinstance(o, dict):
        if {"diff", "ci95", "a", "b"} <= o.keys():
            yield o
        else:
            for v in o.values():
                yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


PAIRED = {(p["a"], p["b"], p["test"], p["view"], p.get("region")): p for p in _walk(R)}


def score(model, test, metric):
    for r in SCORES:
        m = r["eval_id"].replace("ltd26/", "").rsplit("/a", 1)[0].split("@")[0]
        if m == model and r["eval_id"].endswith(f"@{test}.v1") and r["metric"] == metric:
            return publication_scores()[(r["eval_id"].split("@")[0], r["eval_id"].split("@")[1], metric)]
    raise KeyError((model, test, metric))


for m in T:
    for t in TESTS:
        v = T[m][t]
        for view, x in (v.items() if isinstance(v, dict) else [("dialect", v)]):
            macros[f"wer:{m}:{t}:{view}"] = f2(x)
for rec in RECIPES:
    for t in TESTS:
        for view in ("dialect", "standard", "either"):
            try:
                macros[f"mean:{rec}:{t}:{view}"] = f2(mean(seeds(rec, t, view)))
            except (KeyError, TypeError, ZeroDivisionError):
                pass
for (a, b, t, view, region), p in PAIRED.items():
    key = f"diff:{a}-{b}:{t.replace('.v1', '')}:{view}" + (f":{region}" if region else "")
    macros[key] = ci(p, unit=True)
    macros[key + ":d"] = signed(p["diff"], 2) + "\\,pp"
for k, v in R["seed_noise"].items():
    macros[f"n:seedsd:{k}"] = f"{v['pooled_seed_sd']:.2f}\\,pp"
for r, v in R["RQ4_region"].items():
    for k in ("left_out", "all_regions_D80", "all_regions_D50", "no_dialect_S"):
        macros[f"n:region:{r}:{k}"] = f2(v[k])
for m, d in R["RQ5_seen_minus_unseen_either"].items():
    macros[f"n:seen:{m}"] = signed(d) + "\\,pp"
seen = list(R["RQ5_seen_minus_unseen_either"].values())
macros["n:seen:min"], macros["n:seen:max"] = f2(min(seen)), f2(max(seen))
macros["n:seen:above"] = str(sum(1 for d in seen if d > 0))
macros["n:seen:models"] = str(len(seen))
for k, v in STATS.items():
    if isinstance(v, dict) and "hours" in v:
        macros[f"n:{k}:hours"] = f2(v["hours"]); macros[f"n:{k}:clips"] = f"{v['clips']:,}"
        macros[f"n:{k}:speakers"] = str(v["speakers"])
for k, v in STATS["doses"].items():
    macros[f"n:dose:{k}:hours"] = f2(v["hours"]); macros[f"n:dose:{k}:speakers"] = str(v["speakers"])
for k in ("test", "dev", "train"):
    macros[f"n:phone.{k}:hours"] = f2(PHONE["hours"][k]); macros[f"n:phone.{k}:clips"] = f"{PHONE['clips'][k]:,}"
macros["n:phone.excluded:clips"] = str(PHONE["clips"]["excluded"])
macros["n:xspon:hours"] = f2(XSPON["hours"]); macros["n:xspon:clips"] = f"{XSPON['clips']:,}"
macros["n:xspon:radio"] = f2(XSPON["hours_by_source"]["radio"])
macros["n:xspon:dictaphone"] = f2(XSPON["hours_by_source"]["dictaphone"])
hours = {json.load(open(p))["recipe_id"]: json.load(open(p)) for p in (ROOT / "recipes").glob("*.json")}
for rid, rec in hours.items():
    for part in rec["parts"]:
        macros[f"n:recipe:{rid}:{part['code']}:hours"] = f2(part["hours"])
        if part.get("speakers"):
            macros[f"n:recipe:{rid}:{part['code']}:speakers"] = str(part["speakers"])
BENCH = json.load(open(ROOT / "splits" / "v1" / "bench_stats.json"))
macros["n:fleurs:hours"] = f2(BENCH["fleurs.lt.test"]["hours"])
macros["n:cv:hours"] = f2(BENCH["cv.lt.test"]["hours"])
macros["n:cv:speakers"] = str(len({json.loads(l)["speaker"] for l in open(ROOT / "layers" / "cv.lt.test.v1.jsonl")}))
macros["n:dial:speakers:all"] = str(sum(1 for _ in open(ROOT / "data" / "dial_speakers.csv")) - 1)
for name, fn in (("fleurs", "fleurs.lt.test.txt"), ("cv", "cv.lt.test.txt")):
    macros[f"n:{name}:clips"] = f"{len(open(ROOT / 'splits' / 'v1' / fn).read().split()):,}"
sys.path.insert(0, str(ROOT / "code" / "ltd26"))
import score as _score
for test, layer in (("cv.lt.test", "layers/cv.lt.test.v1.jsonl"), ("fleurs.lt.test", "layers/fleurs.lt.test.v1.jsonl")):
    for k in (1, 2):
        ev = next(e for e in csv.DictReader(open(ROOT / "registry" / "evals.csv"))
                  if e["run_id"].startswith(f"ltd26/S/s{k}/") and e["split_id"] == f"{test}.v1")
        refs = _score.load_layer(str(ROOT / layer))
        hyps = {row['id']: row['hyp'] for row in map(json.loads, open(ROOT / ev['hyps_file']))}
        ids = _score.load_split(test + '.v1')
        counts = [_score.edit_distance(_score.normalise(refs[i]['dial']).split(),
                                      _score.normalise(hyps[i]).split()) for i in ids]
        res = {kind: sum(row[j] for row in counts) for j, kind in [(1,'sub'),(2,'del'),(3,'ins')]}
        for kind in ("sub", "del", "ins"):
            macros[f"n:errs:S/s{k}:{test}:{kind}"] = f"{res[kind]:,}"
for tag in ("ending", "pron", "nasal"):
    base = score("S/s1", "dial.test", f"either_err_tag_{tag}") - score("S/s1", "dial.test", "either_err_tag_same")
    for m in ("S+D80.dial/s1", "S+D80.std/s1"):
        gap = score(m, "dial.test", f"either_err_tag_{tag}") - score(m, "dial.test", "either_err_tag_same")
        macros[f"n:gapclosed:{m}:{tag}"] = f"{100 * (base - gap) / base:.2f}"
    macros[f"n:tagratio:S/s1:{tag}"] = f"{score('S/s1', 'dial.test', f'either_err_tag_{tag}') / score('S/s1', 'dial.test', 'either_err_tag_same'):.2f}"
m_s, m_d = mean(seeds("S", "dial.test")), mean(seeds("S+D80.dial", "dial.test"))
macros["n:rel:S+D80.dial-vs-S:dial.test:dialect"] = f"{100 * (m_s - m_d) / m_s:.2f}"
b0 = wer("B/parakeet-tdt-0.6b-v3", "dial.test", "either")
macros["n:rel:S-vs-stock:dial.test:either"] = f"{100 * (b0 - mean(seeds('S', 'dial.test', 'either'))) / b0:.2f}"
for k in (1, 2):
    gd = -PAIRED[(f"S+D80.dial/s{k}", f"S/s{k}", "dial.test.v1", "dialect", None)]["diff"]
    ge = -PAIRED[(f"S+D80.dial/s{k}", f"S/s{k}", "dial.test.v1", "either", None)]["diff"]
    macros[f"n:convshare:s{k}"] = f"{100 * (1 - ge / gd):.2f}"
for m in ("S/s1", "S+X80/s1", "S+D80.dial/s1", "S+D80.std/s1", "D80.std/s1", "B/parakeet-tdt-0.6b-v3"):
    for tag in ("same", "ending", "nasal", "pron", "kept"):
        macros[f"n:tag:{m}:{tag}"] = f2(score(m, "dial.test", f"either_err_tag_{tag}"))
macros["n:nboot"] = f"{R['n_boot']:,}"
macros["n:scorer"] = R["scorer"]
study = [r for r in RUNS if "/smoke/" not in r["run_id"]]
macros["n:runs:completed"] = str(sum(1 for r in study if r["status"] == "completed"))
macros["n:runs:attempts"] = str(len(study))
macros["n:runs:failed"] = str(sum(1 for r in study if r["status"] != "completed"))
macros["n:dial.test:words"] = f"{int(float(next(r['n_words'] for r in SCORES if r['eval_id'].endswith('S/s1/a2@dial.test.v1')))):,}"
macros["n:phone.test:words"] = f"{int(float(next(r['n_words'] for r in SCORES if r['eval_id'].endswith('S/s1/a2@phone.test.v1')))):,}"
d = R["RQ1_dose"]["dial.test_dialect"]
for h, x in zip(["0", "12", "25", "50", "80"], d):
    macros[f"n:dosegain:{h}"] = f"{100 * (d[0] - x) / (d[0] - d[-1]):.2f}"

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "numbers.tex", "w", encoding="utf-8") as f:
    f.write("% generated by analysis/tables.py -- do not edit\n")
    f.write("\\newcommand{\\W}[1]{\\csname ltd:wer:#1\\endcsname\\,\\%}\n"
            "\\newcommand{\\M}[1]{\\csname ltd:mean:#1\\endcsname\\,\\%}\n"
            "\\newcommand{\\Wn}[1]{\\csname ltd:wer:#1\\endcsname}\n"
            "\\newcommand{\\Mn}[1]{\\csname ltd:mean:#1\\endcsname}\n"
            "\\newcommand{\\D}[1]{\\csname ltd:diff:#1\\endcsname}\n"
            "\\newcommand{\\N}[1]{\\csname ltd:n:#1\\endcsname}\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\expandafter\\def\\csname ltd:{k}\\endcsname{{{v}}}\n")


def write(name, lines):
    (OUT / f"{name}.tex").write_text("% generated by analysis/tables.py -- do not edit\n" + "\n".join(lines) + "\n",
                                     encoding="utf-8")
    print(f"paper/generated/{name}.tex")


def cell(rec, test, view="dialect"):
    sv = seeds(rec, test, view)
    return f2(mean(sv)) + ("\\textsuperscript{2}" if len(sv) > 1 else "")


L = ["\\begin{tabular}{l rrr r r rr}", "\\toprule",
     "& \\multicolumn{3}{c}{Dialect test (unseen spk.)} & Seen spk. & Phone & FLEURS & CV\\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}",
     "Model & dial.\\ refs & std.\\ refs & either & either & & & \\\\", "\\midrule"]
for b, lab in zip(BASE, ["parakeet-tdt-0.6b-v3 (stock)", "Whisper large-v3"]):
    L.append(f"{lab} & {f2(wer(b, 'dial.test'))} & {f2(wer(b, 'dial.test', 'standard'))} & "
             f"{f2(wer(b, 'dial.test', 'either'))} & {f2(wer(b, 'dial.seen', 'either'))} & "
             f"{f2(wer(b, 'phone.test'))} & {f2(wer(b, 'fleurs.lt.test'))} & {f2(wer(b, 'cv.lt.test'))}\\\\")
L.append("\\midrule")
groups = [("Dialect-spelling transcripts, by hours", ["S", "S+D12.dial", "S+D25.dial", "S+D50.dial", "S+D80.dial"]),
          ("Control", ["S+X80"]), ("Standard-spelling transcripts", ["S+D25.std", "S+D50.std", "S+D80.std"]),
          ("Dialect data only", ["D80.dial", "D80.std"]),
          ("Region left out", ["S+D80-A.std", "S+D80-D.std", "S+D80-S.std", "S+D80-Z.std"])]
for title, recs in groups:
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}}\\\\")
    for rec in recs:
        L.append(f"\\quad {tex(rec)} & {cell(rec, 'dial.test')} & {cell(rec, 'dial.test', 'standard')} & "
                 f"{cell(rec, 'dial.test', 'either')} & {cell(rec, 'dial.seen', 'either')} & "
                 f"{cell(rec, 'phone.test')} & {cell(rec, 'fleurs.lt.test')} & {cell(rec, 'cv.lt.test')}\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_main", L)

L = ["\\begin{tabular}{l rrrr r}", "\\toprule",
     "Test set, reference view & S & S+D80.dial & S+X80 & S+D80.std & pooled sd\\\\", "\\midrule"]
names = {"dial.test:dialect": "dialect test, dialect refs", "dial.test:either": "dialect test, either form",
         "phone.test:dialect": "phone test", "fleurs.lt.test:dialect": "FLEURS", "cv.lt.test:dialect": "Common Voice"}
for k, v in R["seed_noise"].items():
    L.append(f"{names[k]} & " + " & ".join(signed(x) for x in v["diffs_s1_minus_s2"]) + f" & {v['pooled_seed_sd']:.2f}\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_seeds", L)

TAGS = [("same", "same in both layers"), ("pron", "pronunciation"), ("ending", "ending"), ("nasal", "nasal vowel"),
        ("kept", "dialect word kept")]
models = [("B/parakeet-tdt-0.6b-v3", "stock"), ("S/s1", "S"), ("S+X80/s1", "S+X80"), ("S+D80.dial/s1", "S+D80.dial"),
          ("S+D80.std/s1", "S+D80.std"), ("D80.std/s1", "D80.std")]
L = ["\\begin{tabular}{l " + "r" * len(models) + "}", "\\toprule",
     "Reference word class & " + " & ".join(tex(lab) for _m, lab in models) + "\\\\", "\\midrule"]
for tag, lab in TAGS:
    L.append(f"{lab} & " + " & ".join(f2(score(m, "dial.test", f"either_err_tag_{tag}")) for m, _l in models) + "\\\\")
L.append("\\midrule")
L.append("CER (dialect refs) & " + " & ".join(f2(score(m, "dial.test", "cer_dialect")) for m, _l in models) + "\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_tags", L)

REGION = {"A": "Aukštaitija", "D": "Dzūkija", "S": "Suvalkija", "Z": "Žemaitija"}
L = ["\\begin{tabular}{l r rrrr ll}", "\\toprule",
     "Region & hours & S & S+D80-R.std & S+D50.std & S+D80.std & left-out $-$ D80 & left-out $-$ S\\\\",
     "\\midrule"]
for r, v in R["RQ4_region"].items():
    L.append(f"{REGION[r]} & {macros[f'n:recipe:S+D80-{r}.std:D80-{r}:hours']} & {f2(v['no_dialect_S'])} & "
             f"{f2(v['left_out'])} & {f2(v['all_regions_D50'])} & {f2(v['all_regions_D80'])} & "
             f"{ci(v['vs_D80'])} & {ci(v['vs_S'])}\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_regions", L)

L = ["\\begin{tabular}{l rrrr}", "\\toprule", "Model & Aukštaitija & Dzūkija & Suvalkija & Žemaitija\\\\", "\\midrule"]
for m, lab in [("B/parakeet-tdt-0.6b-v3", "stock"), ("S/s1", "S"), ("S+X80/s1", "S+X80"),
               ("S+D80.dial/s1", "S+D80.dial"), ("S+D80.std/s1", "S+D80.std"), ("D80.dial/s1", "D80.dial")]:
    L.append(f"{tex(lab)} & " + " & ".join(f2(score(m, "dial.test", f"wer_region_{r}")) for r in "ADSZ") + "\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_perregion", L)
write("tab_training", ["\\begin{tabular}{l p{112mm}}", "\\toprule", "Setting & Value\\\\", "\\midrule",
    "Base model & \\texttt{nvidia/parakeet-tdt-0.6b-v3}, full fine-tune, bf16\\\\",
    "Batch & 48 clips per micro-batch ($\\le$ 200 s of audio), 4 accumulation steps (192 clips per update)\\\\",
    "Optimiser & AdamW, $\\beta_1 = 0.9$, $\\beta_2 = 0.98$, weight decay $10^{-3}$\\\\",
    "Learning rate & peak $10^{-4}$, 5\\,\\% linear warm-up, cosine decay to $10^{-6}$\\\\",
    "Augmentation & SpecAugment\\\\", "Clips & 0.5--30 s\\\\",
    "Length & 10{,}000 updates for every run; last checkpoint kept; no early stopping, no checkpoint selection\\\\",
    "Seeds & change data order, augmentation and the initialisation of new layers\\\\",
    "\\bottomrule", "\\end{tabular}"])
L = ["\\begin{tabular}{l r r}", "\\toprule", "Subset & Hours & Speakers\\\\", "\\midrule"]
for k in ("D12", "D25", "D50", "D80"):
    L.append(f"{k} & {f2(STATS['doses'][k]['hours'])} & {STATS['doses'][k]['speakers']}\\\\")
for r, name in (("A", "D80 without Aukštaitija"), ("D", "D80 without Dzūkija"), ("S", "D80 without Suvalkija"),
                ("Z", "D80 without Žemaitija")):
    L.append(f"{name} & {macros[f'n:recipe:S+D80-{r}.std:D80-{r}:hours']} & {macros[f'n:recipe:S+D80-{r}.std:D80-{r}:speakers']}\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_subsets", L)
L = ["\\begin{tabular}{l rrr}", "\\toprule", "Model & pooled WER & mean over speakers & CER\\\\", "\\midrule"]
for m in BASE + [f"{r}/s1" for r in RECIPES] + ["S/s2", "S+D80.dial/s2", "S+X80/s2", "S+D80.std/s2"]:
    L.append(f"{tex(m.replace('B/', ''))} & {f2(score(m, 'dial.test', 'wer_dialect'))} & "
             f"{f2(score(m, 'dial.test', 'wer_mean_over_speakers'))} & {f2(score(m, 'dial.test', 'cer_dialect'))}\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_speakers", L)
print(f"paper/generated/numbers.tex ({len(macros)} macros)")
