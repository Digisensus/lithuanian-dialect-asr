import argparse, collections, difflib, glob, json, os, re, sqlite3, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "data", "std_lexicon.sqlite")
WORD = re.compile(r"^[aąbcčdeęėfghiįyjklmnoprsštuųūvzž]+(\+[aąbcčdeęėfghiįyjklmnoprsštuųūvzž]+)?$")
NASAL = str.maketrans("ąęįų", "aeiu")


def tag(d, s, known):
    if d == s:
        return "same" if known(s) else "kept"
    if s.translate(NASAL) == d or s.translate(NASAL) == d.translate(NASAL):
        return "nasal"
    p = os.path.commonprefix([d, s])
    if len(p) >= 2 and len(d) - len(p) <= 3 and len(s) - len(p) <= 3:
        return "ending"
    return "pron"


def sim(a, b):
    return difflib.SequenceMatcher(None, a.translate(NASAL), b.translate(NASAL)).ratio()


def shifted(dial, std):
    bad = []
    for i, (d, s) in enumerate(zip(dial, std)):
        if d == s:
            continue
        if min(len(d), len(s)) < 4:
            continue
        other = max((sim(dial[j], s) for j in range(len(dial)) if j != i and dial[j] != d), default=0.0)
        if sim(d, s) < 0.4 and other >= 0.8:
            bad.append(i)
    return bad


def warnings(dial, std):
    out = []
    for i, (d, s) in enumerate(zip(dial, std)):
        if d == s:
            continue
        other = max((sim(dial[j], s) for j in range(len(dial)) if j != i and dial[j] != d), default=0.0)
        if any(0 <= j < len(std) and std[j] == s and dial[j] != d for j in (i - 1, i + 1)):
            out.append((i, "repeat"))
        elif min(len(d), len(s)) < 4 and sim(d, s) < 0.4 and other >= 0.8:
            out.append((i, "maybe_moved"))
        elif sim(d, s) < 0.45:
            out.append((i, "big_change"))
    return out


def check_chunk(inp, out, known):
    src = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    errors, good = [], []
    if not os.path.exists(out):
        return [], [{"id": r["id"], "error": "missing output file"} for r in src]
    got = []
    for n, line in enumerate(open(out, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            got.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append({"id": None, "error": f"line {n}: bad JSON ({e.msg})"})
    by_id = collections.Counter(g.get("id") for g in got)
    order = [g.get("id") for g in got if by_id[g.get("id")] == 1]
    if order != [r["id"] for r in src if by_id[r["id"]] == 1]:
        errors.append({"id": None, "error": "ids out of input order"})
    res = {g["id"]: g for g in got if by_id[g.get("id")] == 1}
    for r in src:
        g = res.get(r["id"])
        if by_id[r["id"]] > 1:
            errors.append({"id": r["id"], "error": "duplicate id"}); continue
        if g is None:
            errors.append({"id": r["id"], "error": "missing"}); continue
        std = unicodedata.normalize("NFC", str(g.get("std", ""))).split()
        dial = r["dial"].split()
        unsure = g.get("unsure", [])
        if len(std) != len(dial):
            errors.append({"id": r["id"], "error": f"word count {len(std)} != {len(dial)}"}); continue
        bad = [w for w in std if not WORD.match(w)]
        if bad:
            errors.append({"id": r["id"], "error": f"bad characters in {bad[:3]}"}); continue
        if not isinstance(unsure, list) or any(not isinstance(i, int) or not 0 <= i < len(dial) for i in unsure):
            errors.append({"id": r["id"], "error": f"bad unsure {unsure}"}); continue
        shift = [k for k in shifted(dial, std) if not (isinstance(unsure, list) and k in unsure)]
        if shift:
            errors.append({"id": r["id"], "error": f"words shifted or duplicated at positions {shift}"}); continue
        tags = [tag(d, s, known) for d, s in zip(dial, std)]
        good.append({"id": r["id"], "spk": r["spk"], "dial": r["dial"], "std": " ".join(std),
                     "tags": tags, "unsure": sorted(set(unsure)), "warn": warnings(dial, std)})
    extra = set(res) - {r["id"] for r in src}
    errors += [{"id": i, "error": "id not in input"} for i in sorted(extra)]
    return good, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--chunk")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    cache = {}

    def known(w):
        if w not in cache:
            row = con.execute("select read + spon from lex where word = ?", (w,)).fetchone()
            cache[w] = bool(row and row[0] > 0)
        return cache[w]

    names = [a.chunk] if a.chunk else sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(a.work, "in", "chunk_*.jsonl")))
    all_good, all_err = [], []
    for name in names:
        good, err = check_chunk(os.path.join(a.work, "in", name + ".jsonl"), os.path.join(a.work, "out", name + ".jsonl"), known)
        all_good += good
        all_err += [dict(e, chunk=name) for e in err]
    tags = collections.Counter(t for g in all_good for t in g["tags"])
    ntok = sum(tags.values())
    unknown = collections.Counter(s for g in all_good for d, s, t in zip(g["dial"].split(), g["std"].split(), g["tags"])
                                  if t not in ("same", "kept") and not known(s))
    changes = collections.Counter(f"{d}→{s}" for g in all_good for d, s, t in zip(g["dial"].split(), g["std"].split(), g["tags"])
                                  if t not in ("same", "kept"))
    report = {"chunks": len(names), "clips_ok": len(all_good), "errors": len(all_err),
              "words": ntok, "changed_pct": round(100 * sum(tags[t] for t in ("nasal", "ending", "pron")) / max(1, ntok), 2),
              "tags_pct": {t: round(100 * n / max(1, ntok), 2) for t, n in tags.most_common()},
              "unsure_words_pct": round(100 * sum(len(g["unsure"]) for g in all_good) / max(1, ntok), 2),
              "warnings": dict(collections.Counter(k for g in all_good for _, k in g["warn"])),
              "unknown_std": sum(unknown.values()), "unknown_std_top": unknown.most_common(40),
              "top_changes": changes.most_common(40),
              "error_examples": all_err[:15]}
    if a.merge:
        with open(os.path.join(a.work, "std.jsonl"), "w", encoding="utf-8") as f:
            for g in all_good:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        with open(os.path.join(a.work, "errors.jsonl"), "w", encoding="utf-8") as f:
            for e in all_err:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(os.path.join(a.work, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    if a.chunk:
        report = {k: report[k] for k in ("clips_ok", "errors", "changed_pct", "unknown_std", "unknown_std_top", "error_examples")}
        report["unknown_std_top"] = report["unknown_std_top"][:20]
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(1 if all_err and a.chunk else 0)


if __name__ == "__main__":
    main()
