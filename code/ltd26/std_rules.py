import collections, difflib, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import std_map


def learn(pairs):
    rules = collections.Counter()
    for d, s in pairs:
        dd, ss = f"^{d}$", f"^{s}$"
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, dd, ss, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            old, new = dd[i1:i2], ss[j1:j2]
            left, right = dd[i1 - 1:i1], dd[i2:i2 + 1]
            rules[(left, old, new, right)] += 1
    return {r: c for r, c in rules.items() if c >= 3}


def candidates(w, rules, read, max_edits=2):
    ww = f"^{w}$"
    out = {}
    frontier = {ww: 0.0}
    for _ in range(max_edits):
        nxt = {}
        for form, sc in frontier.items():
            for (l, old, new, r), c in rules.items():
                pat = l + old + r
                start = form.find(pat)
                while start != -1:
                    cand = form[:start] + l + new + r + form[start + len(pat):]
                    s2 = sc + math.log(c)
                    if cand not in nxt or nxt[cand] < s2:
                        nxt[cand] = s2
                    start = form.find(pat, start + 1)
        for cand, sc in nxt.items():
            word = cand.strip("^$")
            if word and read(word) >= 3:
                score = sc + 0.5 * math.log(read(word))
                out[word] = max(out.get(word, -1e9), score)
        frontier = dict(sorted(nxt.items(), key=lambda x: -x[1])[:200])
    return sorted(out.items(), key=lambda x: -x[1])


def evaluate(map_path, split, margin=1.0):
    fmap, read = std_map.load_map(map_path), std_map.lex()
    pairs = [(f, e["std"]) for f, e in fmap.items() if e["decision"] == "std" and "+" not in e["std"]]
    rules = learn(pairs)
    ids = set(open(os.path.join(std_map.SPLITS, f"dial.{split}.txt")).read().split())
    tot = mapped = correct = kept_ok = 0
    ex = []
    seen = set()
    for g in std_map.gold_rows():
        if g["id"] not in ids:
            continue
        for d, s in zip(g["dial"].split(), g["std"].split()):
            if d in fmap or read(d) >= 3 or (d, s) in seen:
                continue
            seen.add((d, s))
            tot += 1
            c = candidates(d, rules, read)
            if c and (len(c) == 1 or c[0][1] - c[1][1] >= margin):
                mapped += 1
                correct += c[0][0] == s
                if c[0][0] != s and len(ex) < 12:
                    ex.append(f"{d}: rules {c[0][0]} | per-clip {s}")
            else:
                kept_ok += d == s
    print(json.dumps({"rules": len(rules), "unmapped_forms": tot, "rule_mapped": mapped,
                      "rule_accuracy_pct": round(100 * correct / max(1, mapped), 1),
                      "coverage_pct": round(100 * mapped / max(1, tot), 1),
                      "left_unchanged_but_correct": kept_ok,
                      "overall_acc_pct": round(100 * (correct + kept_ok) / max(1, tot), 1), "errors": ex},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
