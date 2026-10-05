import argparse, collections, glob, json, os, sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "data", "std_lexicon.sqlite")
STD_TOKENS = 69_561_535
RATIO = 10.0


def lexicon():
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    return lambda w: con.execute("select read, spon from lex where word = ?", (w,)).fetchone() or (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--contexts", type=int, default=3)
    a = ap.parse_args()
    clips = [json.loads(l) for f in sorted(glob.glob(a.clips)) for l in open(f, encoding="utf-8") if l.strip()]
    n = collections.Counter()
    ctx = collections.defaultdict(list)
    regions = collections.defaultdict(collections.Counter)
    for c in clips:
        for w in c["dial"].split():
            n[w] += 1
            regions[w][c["spk"][0]] += 1
            if len(ctx[w]) < a.contexts and c["dial"] not in ctx[w]:
                ctx[w].append(c["dial"])
    lex = lexicon()
    total = sum(n.values())
    out = []
    for w, k in n.most_common():
        read, spon = lex(w)
        ratio = (k / total) / ((read + spon + 1) / STD_TOKENS)
        cls = "nonstd" if read < 3 else ("dialect" if ratio >= RATIO and k >= 3 else None)
        if cls:
            out.append({"type": w, "n": k, "read": read, "spon": spon, "ratio": round(ratio, 1), "class": cls,
                        "regions": dict(regions[w]), "contexts": ctx[w]})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["class"] for r in out)
    print(json.dumps({"clips": len(clips), "tokens": total, "types": len(n), "decision_types": len(out),
                      "by_class": dict(c), "tokens_covered_pct": round(100 * sum(r["n"] for r in out) / total, 1),
                      "dialect_class_examples": [r["type"] for r in out if r["class"] == "dialect"][:40]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
