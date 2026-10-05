import collections, glob, json, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from std_validate import WORD, shifted, tag, DB
from std_policy import RULES

ROOT = os.path.join(HERE, "..", "..")
LAYER = os.path.join(ROOT, "layers", "dial.standard.v1.jsonl")
W = os.path.join(ROOT, "work", "std", "fix")
MAX_CLIPS, MAX_FIX = 150, 300


def silver(split=None):
    for l in open(LAYER, encoding="utf-8"):
        r = json.loads(l)
        if r["tier"] == "silver" and (split is None or r["split"] == split):
            yield r


def agent_rec(r):
    fix = sorted(r["unsure"])
    words = [a[1] for a in r["alignment"]]
    for k in fix:
        words[k] = f"⟨{r['alignment'][k][0]}⟩"
    return {"id": r["id"], "spk": r["id"][:3], "dial": r["dial"], "ctx": " ".join(words), "fix": fix}


def write_chunks(out, recs):
    chunks, cur, n = [], [], 0
    for r in recs:
        if cur and (len(cur) >= MAX_CLIPS or n + len(r["fix"]) > MAX_FIX):
            chunks.append(cur); cur, n = [], 0
        cur.append(r); n += len(r["fix"])
    if cur:
        chunks.append(cur)
    os.makedirs(os.path.join(out, "in"), exist_ok=True)
    os.makedirs(os.path.join(out, "out"), exist_ok=True)
    for k, ch in enumerate(chunks):
        with open(os.path.join(out, "in", f"chunk_{k:03d}.jsonl"), "w", encoding="utf-8") as f:
            for r in ch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"clips": len(recs), "fix_words": sum(len(r["fix"]) for r in recs), "chunks": len(chunks),
            "clips_per_chunk": [len(c) for c in chunks], "fix_per_chunk": [sum(len(r["fix"]) for r in c) for c in chunks]}


def prep(out, id_file):
    ids = set()
    for l in open(id_file, encoding="utf-8"):
        l = l.strip()
        if l:
            ids.add(json.loads(l)["id"] if l.startswith("{") else l)
    recs = [agent_rec(r) for r in silver() if r["id"] in ids and r["unsure"]]
    print(json.dumps(write_chunks(out, recs)))


def plan(split="train", first=301, chunks_per_batch=8):
    parts = [split] + (["excluded"] if split == "train" else [])
    q, batch, chunk_i, clips, n = [], first, 0, 0, 0
    for part in parts:
        for r in silver(part):
            if not r["unsure"]:
                continue
            k = len(r["unsure"])
            if clips and (clips >= MAX_CLIPS or n + k > MAX_FIX):
                chunk_i, clips, n = chunk_i + 1, 0, 0
                if chunk_i == chunks_per_batch:
                    batch, chunk_i = batch + 1, 0
            q.append((batch, r["id"], part))
            clips, n = clips + 1, n + k
    os.makedirs(W, exist_ok=True)
    with open(os.path.join(W, "queue.tsv"), "w") as f:
        for b, i, p in q:
            f.write(f"{b}\t{i}\t{p}\n")
    print(json.dumps({"clips": len(q), "batches": q[-1][0] - first + 1 if q else 0, "first": first}))


def prep_batch(n):
    ids = {i for b, i, _ in (l.rstrip("\n").split("\t") for l in open(os.path.join(W, "queue.tsv"))) if int(b) == n}
    out = os.path.join(W, f"f{n:03d}")
    recs = [agent_rec(r) for r in silver() if r["id"] in ids]
    print(json.dumps(dict(write_chunks(out, recs), batch=n)))


def out_lines(outp):
    parts = [outp] if os.path.exists(outp) else sorted(glob.glob(outp[:-6] + ".p[0-9][0-9].jsonl"))
    return [l for p in parts for l in open(p, encoding="utf-8")] if parts else None


def check_chunk(inp, outp):
    src = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    lines = out_lines(outp)
    if lines is None:
        return [], [{"id": r["id"], "error": "missing output file"} for r in src]
    got, errors = [], []
    for n, line in enumerate(lines, 1):
        if line.strip():
            try:
                got.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append({"id": None, "error": f"line {n}: bad JSON ({e.msg})"})
    cnt = collections.Counter(g.get("id") for g in got)
    if [g.get("id") for g in got if cnt[g.get("id")] == 1] != [r["id"] for r in src if cnt[r["id"]] == 1]:
        errors.append({"id": None, "error": "ids out of input order"})
    res = {g["id"]: g for g in got if cnt[g.get("id")] == 1}
    good = []
    for r in src:
        g = res.get(r["id"])
        if cnt[r["id"]] > 1:
            errors.append({"id": r["id"], "error": "duplicate id"}); continue
        if g is None:
            errors.append({"id": r["id"], "error": "missing"}); continue
        fx, uns = g.get("fix"), g.get("unsure", [])
        if not isinstance(fx, dict) or sorted(fx) != sorted(str(k) for k in r["fix"]):
            errors.append({"id": r["id"], "error": f"fix keys {sorted(fx) if isinstance(fx, dict) else fx} != positions {r['fix']}"}); continue
        bad = [w for w in fx.values() if not isinstance(w, str) or not WORD.match(w)]
        if bad:
            errors.append({"id": r["id"], "error": f"bad word {bad[:3]} (one lowercase word per position, + only in a merge)"}); continue
        if not isinstance(uns, list) or any(k not in r["fix"] for k in uns):
            errors.append({"id": r["id"], "error": f"unsure {uns} must be a subset of fix positions"}); continue
        dial = r["dial"].split()
        std = r["ctx"].split()
        for k in r["fix"]:
            std[k] = fx[str(k)]
        shift = [k for k in shifted(dial, std) if k in r["fix"] and k not in uns]
        if shift:
            errors.append({"id": r["id"], "error": f"word at positions {shift} looks like another position's word"}); continue
        good.append({"id": r["id"], "fix": {int(k): w for k, w in fx.items()}, "unsure": sorted(set(uns))})
    extra = set(res) - {r["id"] for r in src}
    errors += [{"id": i, "error": "id not in input"} for i in sorted(extra)]
    return good, errors


def check(out, name):
    good, err = check_chunk(os.path.join(out, "in", name + ".jsonl"), os.path.join(out, "out", name + ".jsonl"))
    words = sum(len(g["fix"]) for g in good)
    print(json.dumps({"clips_ok": len(good), "errors": len(err), "words": words,
                      "unsure_pct": round(100 * sum(len(g["unsure"]) for g in good) / max(1, words), 1),
                      "error_examples": err[:10]}, ensure_ascii=False, indent=1))
    sys.exit(1 if err else 0)


def merge(out):
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    cache = {}

    def known(w):
        if w not in cache:
            row = con.execute("select read + spon from lex where word = ?", (w,)).fetchone()
            cache[w] = bool(row and row[0] > 0)
        return cache[w]

    dec, errs = {}, []
    for p in sorted(glob.glob(os.path.join(out, "in", "chunk_*.jsonl"))):
        name = os.path.basename(p)[:-6]
        good, err = check_chunk(p, os.path.join(out, "out", name + ".jsonl"))
        dec.update({g["id"]: g for g in good})
        errs += [dict(e, chunk=name) for e in err]
    rules = collections.defaultdict(list)
    for d, cur, new in RULES:
        rules[d].append((cur, new))
    pol, tags, changes = [], collections.Counter(), collections.Counter()
    n = 0
    with open(os.path.join(out, "fixed.jsonl"), "w", encoding="utf-8") as f:
        for r in silver():
            g = dec.get(r["id"])
            if not g:
                continue
            al = [list(a) for a in r["alignment"]]
            for k, w in g["fix"].items():
                for cur, new in rules.get(al[k][0], []):
                    if (cur is None or w in cur) and w != new:
                        pol.append({"id": r["id"], "pos": k, "dial": al[k][0], "from": w, "to": new}); w = new
                if w != al[k][1]:
                    changes[f"{al[k][3]}:{al[k][0]}: {al[k][1]}→{w}"] += 1
                al[k][1], al[k][2], al[k][3] = w, tag(al[k][0], w, known), "F"
                tags[al[k][2]] += 1
            f.write(json.dumps({"id": r["id"], "split": r["split"], "dial": r["dial"],
                                "std": " ".join(a[1].replace("+", " ") for a in al),
                                "src": "".join(a[3] for a in al), "unsure": g["unsure"],
                                "tokens": [a[1] for a in al]}, ensure_ascii=False) + "\n")
            n += 1
    with open(os.path.join(out, "policy.jsonl"), "w", encoding="utf-8") as f:
        for p in pol:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "errors.jsonl"), "w", encoding="utf-8") as f:
        for e in errs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    words = sum(tags.values())
    rep = {"clips": n, "errors": len(errs), "error_examples": errs[:5], "words_decided": words,
           "changed_vs_silver": sum(changes.values()), "policy_fixes": len(pol),
           "unsure_pct": round(100 * sum(len(g["unsure"]) for g in dec.values()) / max(1, words), 1),
           "tags_pct": {t: round(100 * c / max(1, words), 1) for t, c in tags.most_common()},
           "top_changes": changes.most_common(25)}
    json.dump(rep, open(os.path.join(out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def score(out, audit):
    fixed = {g["id"]: g for g in map(json.loads, open(os.path.join(out, "fixed.jsonl"), encoding="utf-8"))}
    aud = {g["id"]: g["std"].split() for g in map(json.loads, open(os.path.join(audit, "std.jsonl"), encoding="utf-8"))}
    tot, before, after = collections.Counter(), collections.Counter(), collections.Counter()
    all_w = all_before = all_after = 0
    spk = collections.defaultdict(lambda: [0, 0])
    ex = []
    for r in silver():
        if r["id"] not in aud:
            continue
        ref, g = aud[r["id"]], fixed.get(r["id"])
        for k, a in enumerate(r["alignment"]):
            new = g["tokens"][k] if g else a[1]
            all_w += 1; all_before += a[1] == ref[k]; all_after += new == ref[k]
            sp = spk[r["id"][:3]]; sp[0] += 1; sp[1] += new == ref[k]
            if g and g["src"][k] == "F":
                key = a[3] + ("*" if a[3] == "R" else "")
                tot[key] += 1; before[key] += a[1] == ref[k]; after[key] += new == ref[k]
                if new != ref[k] and len(ex) < 30:
                    ex.append(f"{r['id']} {k} {a[0]}: silver {a[1]} | fix {new} | audit {ref[k]}")
    adj = json.load(open(os.path.join(audit, "adjudication_sample.json"), encoding="utf-8"))
    v = collections.Counter()
    for x in adj:
        g = fixed.get(x["id"])
        new = g["tokens"][x["pos"]] if g else x["silver"]
        right = {"audit": [x["audit"]], "silver": [x["silver"]], "both/undecidable": [x["audit"], x["silver"]]}.get(x["verdict"], [])
        v[(x["verdict"], "fixed" if g and g["src"][x["pos"]] == "F" else "untouched", new in right)] += 1
    n_ok = sum(c for (vd, _, ok), c in v.items() if ok)
    n_dec = sum(c for (vd, _, ok), c in v.items() if vd != "neither")
    words = sum(tot.values())
    res = {"audit_clips": len(aud), "fixed_clips": sum(1 for i in fixed if i in aud),
           "fixed_words": words,
           "fixed_words_agreement_pct": {"silver_before": round(100 * sum(before.values()) / max(1, words), 1),
                                         "after_fix": round(100 * sum(after.values()) / max(1, words), 1)},
           "by_source": {s: {"words": tot[s], "before_pct": round(100 * before[s] / tot[s], 1),
                             "after_pct": round(100 * after[s] / tot[s], 1)} for s in sorted(tot)},
           "all_words_agreement_pct": {"silver_before": round(100 * all_before / all_w, 2), "after_fix": round(100 * all_after / all_w, 2)},
           "adjudicated_300": {"correct_after_fix": n_ok, "decidable": n_dec,
                               "detail": {f"{a}|{b}|{'ok' if c else 'wrong'}": n for (a, b, c), n in sorted(v.items())}},
           "examples_disagree": ex}
    json.dump(res, open(os.path.join(out, "pilot_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if cmd == "prep":
        prep(a[0], opt("--ids"))
    elif cmd == "plan":
        plan(opt("--split", "train"), int(opt("--first", 301)))
    elif cmd == "prep-batch":
        prep_batch(int(a[0]))
    elif cmd == "check":
        check(a[0], a[1])
    elif cmd == "merge":
        merge(a[0])
    elif cmd == "score":
        score(a[0], a[1])
