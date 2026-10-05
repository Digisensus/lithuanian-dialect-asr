import difflib, json, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "data", "std_lexicon.sqlite")


def main():
    d = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    known = lambda w: (con.execute("select read + spon from lex where word = ?", (w,)).fetchone() or (0,))[0] > 0
    out, seen = [], set()
    for l in open(os.path.join(d, "std.jsonl"), encoding="utf-8"):
        g = json.loads(l)
        for dw, sw in zip(g["dial"].split(), g["std"].split()):
            if dw == sw or "+" in sw or sw in seen or known(sw) or len(sw) < 5:
                continue
            seen.add(sw)
            cands = [w for (w,) in con.execute("select word from lex where word like ? and read >= 20", (sw[:3] + "%",))]
            best = difflib.get_close_matches(sw, cands, n=1, cutoff=0.85)
            if best:
                out.append({"id": g["id"], "dial": dw, "std": sw, "suggest": best[0]})
    print(json.dumps({"near_misses": len(out), "examples": out[:top]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
