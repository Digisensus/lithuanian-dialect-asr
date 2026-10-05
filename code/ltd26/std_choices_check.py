import glob, json, os, sys


def main():
    d = sys.argv[1]
    names = [sys.argv[2]] if len(sys.argv) > 2 else sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(d, "in", "part_*.jsonl")))
    errs, n = [], 0
    for name in names:
        items = [json.loads(l) for l in open(os.path.join(d, "in", name + ".jsonl"), encoding="utf-8") if l.strip()]
        path = os.path.join(d, "out", name + ".jsonl")
        if not os.path.exists(path):
            errs.append(f"{name}: missing output"); continue
        got = {}
        for i, l in enumerate(open(path, encoding="utf-8"), 1):
            if l.strip():
                try:
                    g = json.loads(l); got[g["id"]] = g
                except (json.JSONDecodeError, KeyError):
                    errs.append(f"{name} line {i}: bad JSON")
        for it in items:
            n += 1
            g = got.get(it["id"])
            if g is None:
                errs.append(f"{it['id']}: missing"); continue
            c = g.get("c")
            if not isinstance(c, list) or len(c) != len(it["o"]) or any(not isinstance(x, int) or not 1 <= x <= len(o) for x, o in zip(c, it["o"])):
                errs.append(f"{it['id']}: need {len(it['o'])} choices within the option ranges, got {c}")
    print(json.dumps({"items": n, "errors": len(errs), "error_examples": errs[:12]}, ensure_ascii=False))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
