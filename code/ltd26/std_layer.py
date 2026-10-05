import collections, glob, json, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from std_validate import tag, DB

ROOT = os.path.join(HERE, "..", "..")


def main():
    silver_f, out = sys.argv[1], sys.argv[2]
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    cache = {}

    def known(w):
        if w not in cache:
            row = con.execute("select read + spon from lex where word = ?", (w,)).fetchone()
            cache[w] = bool(row and row[0] > 0)
        return cache[w]

    split = {}
    for part in ("test", "dev", "seen", "train", "excluded"):
        for i in open(os.path.join(ROOT, "splits", "v1", f"dial.{part}.txt")).read().split():
            split[i] = part
    n = collections.Counter()
    src_tot = collections.Counter()
    seen = set()
    with open(out, "w", encoding="utf-8") as f:
        def emit(cid, dial, std, src, unsure, tier):
            d, s = dial.split(), std.split()
            assert len(d) == len(s) == len(src), cid
            al = [[a, b, tag(a, b, known), c] for a, b, c in zip(d, s, src)]
            f.write(json.dumps({"id": cid, "split": split.get(cid), "tier": tier, "dial": dial,
                                "std": " ".join(w.replace("+", " ") for w in s), "alignment": al,
                                "unsure": unsure}, ensure_ascii=False) + "\n")
            n[(tier, split.get(cid))] += 1
            src_tot.update(src)
            seen.add(cid)

        for p in sorted(glob.glob(os.path.join(ROOT, "work", "std", "batches", "b*", "std.jsonl"))):
            meta = os.path.join(os.path.dirname(p), "meta.json")
            letter = "P"
            for l in open(p, encoding="utf-8"):
                g = json.loads(l)
                tier = "gold" if split.get(g["id"]) in ("test", "dev", "seen") else "perclip"
                emit(g["id"], g["dial"], g["std"], letter * len(g["dial"].split()), g.get("unsure", []), tier)
        import std_policy
        read = lambda w: (con.execute("select read from lex where word = ?", (w,)).fetchone() or (0,))[0]
        pol = collections.Counter()
        for l in open(silver_f, encoding="utf-8"):
            g = json.loads(l)
            if g["id"] in seen:
                continue
            d, s, unsure = g["dial"].split(), g["std"].split(), set(g.get("unsure", []))
            for k, (dw, sw) in enumerate(zip(d, s)):
                new = sw
                r8 = std_policy.v08(dw, new, read, lambda w: known(w))
                if r8:
                    new = r8[0]
                for rd, cur, nw in std_policy.RULES:
                    if rd != "vė" and dw == rd and (cur is None or new in cur) and not (dw == new and k in unsure):
                        new = nw
                if new != sw:
                    pol[f"{sw}→{new}"] += 1
                    s[k] = new
            emit(g["id"], g["dial"], " ".join(s), g["src"], g.get("unsure", []), "silver")
        print(json.dumps({"silver_policy_changes": sum(pol.values()), "top": pol.most_common(12)}, ensure_ascii=False))
    words = sum(src_tot.values())
    print(json.dumps({"clips": len(seen), "by_tier_split": {f"{t}/{s}": c for (t, s), c in sorted(n.items())},
                      "word_sources_pct": {k: round(100 * v / words, 2) for k, v in src_tot.most_common()}}))


if __name__ == "__main__":
    main()
