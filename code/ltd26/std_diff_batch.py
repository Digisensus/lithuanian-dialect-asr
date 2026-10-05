import collections, glob, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from std_validate import WORD, shifted

ROOT = os.path.join(HERE, "..", "..")
LAYER = os.path.join(ROOT, "layers", "dial.standard.v1.jsonl")
MAX_CLIPS, MAX_NS = 150, 300


def prep(out, id_file):
    import sqlite3
    from std_batch import DB
    ids = []
    for l in open(id_file, encoding="utf-8"):
        l = l.strip()
        if l:
            ids.append(json.loads(l)["id"] if l.startswith("{") else l)
    want = set(ids)
    rows = [json.loads(l) for l in open(LAYER, encoding="utf-8")]
    rows = [r for r in rows if r["id"] in want]
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    read = {}

    def ns(words):
        o = []
        for k, w in enumerate(words):
            if w not in read:
                x = con.execute("select read from lex where word = ?", (w,)).fetchone()
                read[w] = x[0] if x else 0
            if read[w] < 3:
                o.append(k)
        return o

    recs = [{"id": r["id"], "spk": r["id"][:3], "dial": r["dial"], "ns": ns(r["dial"].split())} for r in rows]
    chunks, cur, n = [], [], 0
    for r in recs:
        if cur and (len(cur) >= MAX_CLIPS or n + len(r["ns"]) > MAX_NS):
            chunks.append(cur); cur, n = [], 0
        cur.append(r); n += len(r["ns"])
    if cur:
        chunks.append(cur)
    os.makedirs(os.path.join(out, "in"), exist_ok=True)
    os.makedirs(os.path.join(out, "out", "parts"), exist_ok=True)
    for k, ch in enumerate(chunks):
        with open(os.path.join(out, "in", f"chunk_{k:03d}.jsonl"), "w", encoding="utf-8") as f:
            for r in ch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"clips": len(recs), "chunks": len(chunks), "clips_per_chunk": [len(c) for c in chunks],
                      "ns_per_chunk": [sum(len(r["ns"]) for r in c) for c in chunks]}))


def read_parts(out, name):
    parts = sorted(glob.glob(os.path.join(out, "out", "parts", f"{name}.p[0-9][0-9].jsonl")))
    return [l for p in parts for l in open(p, encoding="utf-8") if l.strip()] if parts else None


def short_shift(dial, std):
    import difflib
    N = str.maketrans("ąęįųūėy", "aeiuuei")
    sim = lambda a, b: difflib.SequenceMatcher(None, a.translate(N), b.translate(N)).ratio()
    bad = []
    for i, (d, s) in enumerate(zip(dial, std)):
        if d == s or len(d) >= 4 or len(s) < 4:
            continue
        near = max((sim(dial[j], s.replace("+", "")) for j in (i - 1, i + 1) if 0 <= j < len(dial) and dial[j] != d), default=0)
        if sim(d, s.replace("+", "")) < 0.4 and near >= 0.75:
            bad.append(i)
    return bad


def neighbour_slip(dial, std):
    import difflib
    N = str.maketrans("ąęįųūėy", "aeiuuei")
    sim = lambda a, b: difflib.SequenceMatcher(None, a.translate(N), b.translate(N)).ratio()
    bad = []
    for k, (d, s) in enumerate(zip(dial, std)):
        w = s.replace("+", "")
        if d == s or len(w) <= 3 or sim(d, w) >= 0.5:
            continue
        if any(0 <= j < len(dial) and std[j] == dial[j] and dial[j] != d and sim(dial[j], w) >= 0.75 for j in (k - 1, k + 1)):
            bad.append(k)
    return bad


def check_chunk(out, name):
    src = [json.loads(l) for l in open(os.path.join(out, "in", name + ".jsonl"), encoding="utf-8") if l.strip()]
    lines = read_parts(out, name)
    if lines is None:
        return [], [{"id": r["id"], "error": "no block files"} for r in src]
    got, errors = [], []
    for n, line in enumerate(lines, 1):
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
        dial = r["dial"].split()
        ch, uns = g.get("ch"), g.get("unsure", [])
        if not isinstance(ch, dict) or any(not str(k).isdigit() or not 0 <= int(k) < len(dial) for k in ch):
            errors.append({"id": r["id"], "error": f"ch must map positions 0..{len(dial) - 1} to words"}); continue
        bad = [w for w in ch.values() if not isinstance(w, str) or not WORD.match(w)]
        if bad:
            errors.append({"id": r["id"], "error": f"bad word {bad[:3]} (one lowercase word per position, + only in a merge)"}); continue
        same = [k for k, w in ch.items() if w == dial[int(k)]]
        if same:
            errors.append({"id": r["id"], "error": f"positions {same} list an unchanged word; list only changes"}); continue
        if not isinstance(uns, list) or any(not isinstance(k, int) or not 0 <= k < len(dial) for k in uns):
            errors.append({"id": r["id"], "error": f"bad unsure {uns}"}); continue
        std = list(dial)
        for k, w in ch.items():
            std[int(k)] = w
        slip = neighbour_slip(dial, std)
        if slip:
            errors.append({"id": r["id"], "error": f"change at positions {slip} matches the unchanged neighbouring word; recount the positions"}); continue
        shift = [k for k in set(shifted(dial, std)) | set(short_shift(dial, std)) if k not in uns]
        if shift:
            errors.append({"id": r["id"], "error": f"word at positions {shift} looks like another position's word"}); continue
        good.append({"id": r["id"], "std": " ".join(std), "unsure": sorted(set(uns)), "n_ch": len(ch)})
    extra = set(res) - {r["id"] for r in src}
    errors += [{"id": i, "error": "id not in input"} for i in sorted(extra)]
    return good, errors


def check(out, name):
    good, err = check_chunk(out, name)
    print(json.dumps({"clips_ok": len(good), "errors": len(err), "changed_words": sum(g["n_ch"] for g in good),
                      "error_examples": err[:10]}, ensure_ascii=False, indent=1))
    sys.exit(1 if err else 0)


def merge(out):
    full = os.path.join(out, "full")
    if os.path.exists(full):
        shutil.rmtree(full)
    shutil.copytree(os.path.join(out, "in"), os.path.join(full, "in"))
    os.makedirs(os.path.join(full, "out"))
    errs = []
    for p in sorted(glob.glob(os.path.join(out, "in", "chunk_*.jsonl"))):
        name = os.path.basename(p)[:-6]
        good, err = check_chunk(out, name)
        errs += err
        with open(os.path.join(full, "out", name + ".jsonl"), "w", encoding="utf-8") as f:
            for g in good:
                f.write(json.dumps({"id": g["id"], "std": g["std"], "unsure": g["unsure"]}, ensure_ascii=False) + "\n")
    pol = subprocess.run([sys.executable, os.path.join(HERE, "std_policy.py"), full], capture_output=True, text=True).stdout.strip()
    subprocess.run([sys.executable, os.path.join(HERE, "std_validate.py"), full, "--merge"], capture_output=True, text=True)
    rep = json.load(open(os.path.join(full, "report.json"), encoding="utf-8"))
    print(json.dumps({"diff_errors": len(errs), "diff_error_examples": errs[:5], "policy": pol,
                      **{k: rep[k] for k in ("clips_ok", "errors", "words", "changed_pct", "tags_pct", "unsure_words_pct", "unknown_std")}},
                     ensure_ascii=False, indent=1))


def expand(d):
    if os.path.exists(os.path.join(d, "fixes.jsonl")) and "--force" not in sys.argv:
        sys.exit(f"{d} has fixes.jsonl (policy/manual fixes on out/): re-expanding would drop them; use --force and re-apply")
    rep = {}
    for p in sorted(glob.glob(os.path.join(d, "in", "chunk_*.jsonl"))):
        name = os.path.basename(p)[:-6]
        good, err = check_chunk(d, name)
        with open(os.path.join(d, "out", name + ".jsonl"), "w", encoding="utf-8") as f:
            for g in good:
                f.write(json.dumps({"id": g["id"], "std": g["std"], "unsure": g["unsure"]}, ensure_ascii=False) + "\n")
        rep[name] = {"clips_ok": len(good), "errors": len(err)}
    print(json.dumps(rep))


def start(n):
    code = subprocess.run([sys.executable, os.path.join(HERE, "std_batch.py"), "prep", str(n)], capture_output=True, text=True)
    print(code.stdout.strip() or code.stderr.strip())
    d = os.path.join(ROOT, "work", "std", "batches", f"b{n:03d}")
    os.makedirs(os.path.join(d, "out", "parts"), exist_ok=True)
    json.dump({"model": "reviewer",
               "output": "changed words only, expanded by std_diff_batch.py expand"}, open(os.path.join(d, "meta.json"), "w"), indent=1)


def score(out, audit):
    new = {g["id"]: g["std"].split() for g in map(json.loads, open(os.path.join(out, "full", "std.jsonl"), encoding="utf-8"))}
    aud = {g["id"]: g["std"].split() for g in map(json.loads, open(os.path.join(audit, "std.jsonl"), encoding="utf-8"))}
    lay = {}
    for l in open(LAYER, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in new:
            lay[r["id"]] = r
    tot = agree = ch_tot = ch_agree = sil_agree = 0
    by_src = collections.defaultdict(lambda: [0, 0, 0])
    spk = collections.defaultdict(lambda: [0, 0])
    ex = []
    for i, hyp in new.items():
        ref, al = aud[i], lay[i]["alignment"]
        for k, (d, s, _, src) in enumerate(al):
            h, y = hyp[k], ref[k]
            tot += 1; agree += h == y; sil_agree += s == y
            b = by_src[src]; b[0] += 1; b[1] += h == y; b[2] += s == y
            sp = spk[i[:3]]; sp[0] += 1; sp[1] += h == y
            if h != d or y != d:
                ch_tot += 1; ch_agree += h == y
            if h != y and len(ex) < 30:
                ex.append(f"{i} {k} {d}: new {h} | audit {y} | silver {s}")
    adj = json.load(open(os.path.join(audit, "adjudication_sample.json"), encoding="utf-8"))
    v = collections.Counter()
    for x in adj:
        if x["id"] not in new:
            continue
        h = new[x["id"]][x["pos"]]
        right = {"audit": [x["audit"]], "silver": [x["silver"]], "both/undecidable": [x["audit"], x["silver"]]}.get(x["verdict"], [])
        v[x["verdict"], h in right] += 1
    res = {"clips": len(new), "words": tot,
           "agreement_with_audit_pct": round(100 * agree / tot, 2),
           "silver_agreement_with_audit_pct": round(100 * sil_agree / tot, 2),
           "changed_words": ch_tot, "changed_word_agreement_pct": round(100 * ch_agree / max(1, ch_tot), 1),
           "by_silver_source": {s: {"words": b[0], "new_vs_audit_pct": round(100 * b[1] / b[0], 1),
                                    "silver_vs_audit_pct": round(100 * b[2] / b[0], 1)} for s, b in sorted(by_src.items())},
           "adjudicated_300": {"new_correct": sum(c for (vd, ok), c in v.items() if ok),
                               "decidable": sum(c for (vd, ok), c in v.items() if vd != "neither"),
                               "detail": {f"{a}|{'ok' if b else 'wrong'}": c for (a, b), c in sorted(v.items())}},
           "examples_disagree": ex}
    json.dump(res, open(os.path.join(out, "pilot_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if cmd == "prep":
        prep(a[0], opt("--ids"))
    elif cmd == "check":
        check(a[0], a[1])
    elif cmd == "merge":
        merge(a[0])
    elif cmd == "score":
        score(a[0], a[1])
    elif cmd == "expand":
        expand(a[0])
    elif cmd == "start":
        start(int(a[0]))
