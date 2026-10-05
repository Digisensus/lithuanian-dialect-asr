import collections, glob, json, os, re, sys

WORD = re.compile(r"^[aąbcčdeęėfghiįyjklmnoprsštuųūvzž]+(\+[aąbcčdeęėfghiįyjklmnoprsštuųūvzž]+)?$")


def window(sentence, w, k=6):
    ws = sentence.split()
    i = ws.index(w) if w in ws else 0
    left = ("… " if i - k > 0 else "") + " ".join(ws[max(0, i - k):i])
    right = " ".join(ws[i + 1:i + 1 + k]) + (" …" if i + 1 + k < len(ws) else "")
    return f"{left} ⟨{w}⟩ {right}".strip()


def export(types_f, out_dir, parts=3):
    rows = [json.loads(l) for l in open(types_f, encoding="utf-8")]
    os.makedirs(os.path.join(out_dir, "in"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)
    size = -(-len(rows) // parts)
    for p in range(parts):
        with open(os.path.join(out_dir, "in", f"part_{p:02d}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows[p * size:(p + 1) * size]:
                reg = " ".join(f"{k}{v}" for k, v in sorted(r["regions"].items(), key=lambda x: -x[1]))
                f.write(json.dumps({"t": r["type"], "n": r["n"], "reg": reg,
                                    "ctx": [window(c, r["type"]) for c in r["contexts"]]}, ensure_ascii=False) + "\n")
    print(json.dumps({"forms": len(rows), "parts": parts}))


def check(out_dir, part=None):
    errors, n, dec = [], 0, collections.Counter()
    names = [part] if part else sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(out_dir, "in", "part_*.jsonl")))
    for name in names:
        want = [json.loads(l)["t"] for l in open(os.path.join(out_dir, "in", name + ".jsonl"), encoding="utf-8")]
        path = os.path.join(out_dir, "out", name + ".jsonl")
        if not os.path.exists(path):
            errors.append(f"{name}: missing output"); continue
        got = {}
        for i, l in enumerate(open(path, encoding="utf-8"), 1):
            if not l.strip():
                continue
            try:
                g = json.loads(l)
            except json.JSONDecodeError:
                errors.append(f"{name} line {i}: bad JSON"); continue
            t, d = g.get("t"), g.get("d")
            if t in got:
                errors.append(f"{name}: duplicate {t}")
            got[t] = g
            if d not in ("std", "keep", "ctx", "merge"):
                errors.append(f"{name}: {t}: bad decision {d!r}")
            elif d in ("std", "merge") and not (isinstance(g.get("s"), str) and WORD.match(g["s"]) and ("+" in g["s"]) == (d == "merge")):
                errors.append(f"{name}: {t}: bad s {g.get('s')!r}")
            elif d == "ctx" and not (isinstance(g.get("c"), list) and len(g["c"]) >= 2 and all(isinstance(x, str) and WORD.match(x) for x in g["c"])):
                errors.append(f"{name}: {t}: ctx needs >= 2 candidates, got {g.get('c')!r}")
            dec[d] += 1
        missing = [t for t in want if t not in got]
        extra = [t for t in got if t not in set(want)]
        if missing:
            errors.append(f"{name}: {len(missing)} forms missing, e.g. {missing[:5]}")
        if extra:
            errors.append(f"{name}: {len(extra)} forms not in input, e.g. {extra[:5]}")
        n += len(want)
    print(json.dumps({"forms": n, "decisions": dict(dec), "errors": len(errors), "error_examples": errors[:15]}, ensure_ascii=False))
    return not errors


def merge(out_dir, typemap_f):
    with open(typemap_f, "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(os.path.join(out_dir, "out", "part_*.jsonl"))):
            for l in open(p, encoding="utf-8"):
                if not l.strip():
                    continue
                g = json.loads(l)
                d = {"type": g["t"], "decision": g["d"], "unsure": bool(g.get("u"))}
                if g["d"] in ("std", "merge"):
                    d["std"] = g["s"]
                if g["d"] == "ctx":
                    d["cands"] = list(dict.fromkeys(g["c"]))
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export(sys.argv[2], sys.argv[3], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    elif cmd == "check":
        sys.exit(0 if check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) else 1)
    elif cmd == "merge":
        merge(sys.argv[2], sys.argv[3])
