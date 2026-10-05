import collections, glob, json, os, sqlite3, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DB = os.path.join(ROOT, "data", "std_lexicon.sqlite")
BATCHES = os.path.join(ROOT, "work", "std", "batches")
SPLITS = os.path.join(ROOT, "splits", "v1")


def lex():
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    cache = {}

    def read(w):
        if w not in cache:
            cache[w] = (con.execute("select read from lex where word = ?", (w,)).fetchone() or (0,))[0]
        return cache[w]
    return read


def gold_rows(exclude=None):
    skip = set(open(os.path.join(SPLITS, f"dial.{exclude}.txt")).read().split()) if exclude else set()
    for f in sorted(glob.glob(os.path.join(BATCHES, "b*", "std.jsonl"))):
        for l in open(f, encoding="utf-8"):
            g = json.loads(l)
            if g["id"] not in skip:
                yield g


def build(out, exclude=None):
    m = collections.defaultdict(collections.Counter)
    for g in gold_rows(exclude):
        for d, s in zip(g["dial"].split(), g["std"].split()):
            m[d][s] += 1
    n_dec = collections.Counter()
    with open(out, "w", encoding="utf-8") as f:
        for form, c in sorted(m.items(), key=lambda x: -sum(x[1].values())):
            n = sum(c.values())
            top, k = c.most_common(1)[0]
            share = k / n
            dec = "keep" if top == form and share >= 0.9 else ("std" if share >= 0.9 else "ctx")
            n_dec[dec] += 1
            f.write(json.dumps({"form": form, "n": n, "out": dict(c), "decision": dec, "std": top,
                                "share": round(share, 3)}, ensure_ascii=False) + "\n")
    print(json.dumps({"forms": len(m), "decisions": dict(n_dec), "excluded": exclude}))


def load_map(path):
    return {e["form"]: e for e in map(json.loads, open(path, encoding="utf-8"))}


NASAL = {"a": "ą", "e": "ę", "i": "į", "u": "ų"}


def convert(words, fmap, read, reviewed=None):
    std, src = [], []
    for w in words:
        r = reviewed.get(w) if reviewed else None
        e = fmap.get(w)
        if r:
            dec = r["decision"]
            std.append(r["std"] if dec in ("std", "merge") else (r.get("cands") or [w])[0] if dec == "ctx" else w)
            src.append("r" if dec == "ctx" else "R")
        elif e:
            std.append(e["std"])
            src.append("C" if e["decision"] == "ctx" else "M")
        elif read(w) >= 3:
            n = w[:-1] + NASAL[w[-1]] if w[-1:] in NASAL else None
            if n and read(n) >= 5 * max(1, read(w)):
                std.append(n); src.append("N")
            else:
                std.append(w); src.append("S")
        else:
            std.append(w); src.append("U")
    return std, src


def load_reviewed(path):
    return {d["type"]: d for d in map(json.loads, open(path, encoding="utf-8"))} if path else None


def apply(map_path, reviewed_path, out, split="train"):
    import pyarrow.parquet as pq
    fmap, read, rev = load_map(map_path), lex(), load_reviewed(reviewed_path)
    done = {g["id"] for g in gold_rows()}
    train = set(open(os.path.join(SPLITS, f"dial.{split}.txt")).read().split()) - done
    tags = collections.Counter()
    with open(out, "w", encoding="utf-8") as f:
        for pf in sorted(glob.glob(os.path.join(ROOT, "..", "lithuanian-dialect-speech-liepa-3-100h-punctuated", "data", "*.parquet"))):
            for r in pq.read_table(pf, columns=["id", "speaker_id", "text_normalized"]).to_pylist():
                if r["id"] not in train:
                    continue
                d = r["text_normalized"].split()
                s, src = convert(d, fmap, read, rev)
                unsure = [k for k, t in enumerate(src) if t in "CrU" or (t == "R" and rev[d[k]].get("unsure"))]
                tags.update(src)
                f.write(json.dumps({"id": r["id"], "spk": r["speaker_id"], "dial": " ".join(d), "std": " ".join(s),
                                    "src": "".join(src), "unsure": unsure}, ensure_ascii=False) + "\n")
    n = sum(tags.values())
    print(json.dumps({"clips": len(train), "words": n, "sources_pct": {t: round(100 * c / n, 2) for t, c in tags.most_common()}}))


def review_list(map_path, out_dir, min_n=1, contexts=2, split="train", reviewed_path=None):
    import pyarrow.parquet as pq
    fmap, read, rev = load_map(map_path), lex(), load_reviewed(reviewed_path) or {}
    done = {g["id"] for g in gold_rows()}
    for f in glob.glob(os.path.join(BATCHES, "b*", "in", "chunk_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            done.add(json.loads(l)["id"])
    train = set(open(os.path.join(SPLITS, f"dial.{split}.txt")).read().split()) - done
    n, ctx, reg = collections.Counter(), collections.defaultdict(list), collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(os.path.join(ROOT, "..", "lithuanian-dialect-speech-liepa-3-100h-punctuated", "data", "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "text_normalized", "speaker_id"]).to_pylist():
            if r["id"] not in train:
                continue
            ws = r["text_normalized"].split()
            for k, w in enumerate(ws):
                if w in fmap or w in rev or read(w) >= 3:
                    continue
                n[w] += 1
                reg[w][r["speaker_id"][0]] += 1
                if len(ctx[w]) < contexts:
                    left = " ".join(ws[max(0, k - 5):k])
                    right = " ".join(ws[k + 1:k + 6])
                    ctx[w].append(f"{left} ⟨{w}⟩ {right}".strip())
    forms = [w for w, k in n.most_common() if k >= min_n]
    os.makedirs(os.path.join(out_dir, "in"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)
    with open(os.path.join(out_dir, "all_forms.jsonl"), "w", encoding="utf-8") as f:
        for w in forms:
            f.write(json.dumps({"t": w, "n": n[w], "reg": " ".join(f"{a}{b}" for a, b in reg[w].most_common()),
                                "ctx": ctx[w]}, ensure_ascii=False) + "\n")
    print(json.dumps({"clips": len(train), "forms_to_review": len(forms),
                      "by_n": {"1": sum(1 for w in forms if n[w] == 1), "2": sum(1 for w in forms if n[w] == 2),
                               ">=3": sum(1 for w in forms if n[w] >= 3)}}))


def review_parts(all_forms, out_root, per_part=600, parts_per_batch=8):
    forms = [l for l in open(all_forms, encoding="utf-8") if l.strip()]
    k = 0
    for start in range(0, len(forms), per_part):
        b, p = k // parts_per_batch + 1, k % parts_per_batch
        d = os.path.join(out_root, f"rb{b:03d}")
        os.makedirs(os.path.join(d, "in"), exist_ok=True)
        os.makedirs(os.path.join(d, "out"), exist_ok=True)
        with open(os.path.join(d, "in", f"part_{p:02d}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(forms[start:start + per_part])
        k += 1
    print(json.dumps({"forms": len(forms), "parts": k, "batches": -(-k // parts_per_batch), "per_part": per_part}))


def evaluate(map_path, split):
    fmap, read = load_map(map_path), lex()
    ids = set(open(os.path.join(SPLITS, f"dial.{split}.txt")).read().split())
    tot, agree = collections.Counter(), collections.Counter()
    clips = same = 0
    ex = collections.defaultdict(list)
    for g in gold_rows():
        if g["id"] not in ids:
            continue
        d, ref = g["dial"].split(), g["std"].split()
        hyp, src = convert(d, fmap, read)
        clips += 1
        same += hyp == ref
        for dw, h, r, s in zip(d, hyp, ref, src):
            tot[s] += 1
            agree[s] += h == r
            if h != r and len(ex[s]) < 8:
                ex[s].append(f"{dw}: map {h} | per-clip {r}")
    words = sum(tot.values())
    print(json.dumps({"split": split, "clips": clips, "clip_exact_pct": round(100 * same / max(1, clips), 1),
                      "word_accuracy_pct": round(100 * sum(agree.values()) / max(1, words), 2),
                      "by_source": {s: {"words": tot[s], "share_pct": round(100 * tot[s] / words, 1),
                                        "accuracy_pct": round(100 * agree[s] / tot[s], 1)} for s in sorted(tot)},
                      "examples": dict(ex)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if cmd == "build":
        build(a[0], opt("--exclude-split"))
    elif cmd == "review-list":
        review_list(a[0], a[1], int(opt("--min-n", 1)), split=opt("--split", "train"), reviewed_path=opt("--reviewed"))
    elif cmd == "eval":
        evaluate(a[0], a[1])
    elif cmd == "apply":
        apply(a[0], opt("--reviewed"), a[1], opt("--split", "train"))
    elif cmd == "review-parts":
        review_parts(a[0], a[1], int(opt("--per-part", 600)))
