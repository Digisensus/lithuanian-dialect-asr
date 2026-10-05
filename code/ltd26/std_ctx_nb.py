import collections, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import std_map


def feats(ws, k):
    f = ["b"]
    p = ws[k - 1] if k > 0 else "<s>"
    n = ws[k + 1] if k + 1 < len(ws) else "</s>"
    n2 = ws[k + 2] if k + 2 < len(ws) else "</s>"
    f += [f"p={p}", f"n={n}", f"ps={p[-2:]}", f"ns={n[-2:]}", f"ns1={n[-1:]}", f"n2={n2}", f"pos={'first' if k == 0 else 'last' if k == len(ws) - 1 else 'mid'}"]
    return f


def train(map_path, out, exclude=None, min_ex=8):
    fmap = std_map.load_map(map_path)
    ctx = {w for w, e in fmap.items() if e["decision"] == "ctx"}
    data = collections.defaultdict(list)
    for g in std_map.gold_rows(exclude):
        d, s = g["dial"].split(), g["std"].split()
        for k, w in enumerate(d):
            if w in ctx:
                data[w].append((feats(d, k), s[k]))
    model = {}
    for w, ex in data.items():
        if len(ex) < min_ex:
            continue
        prior = collections.Counter(y for _, y in ex)
        cnt = {y: collections.Counter() for y in prior}
        for f, y in ex:
            cnt[y].update(f)
        vocab = {x for c in cnt.values() for x in c}
        model[w] = {"prior": dict(prior), "cnt": {y: dict(c) for y, c in cnt.items()}, "V": len(vocab)}
    json.dump(model, open(out, "w"), ensure_ascii=False)
    print(json.dumps({"ctx_forms": len(ctx), "modelled": len(model), "examples": sum(len(v) for v in data.values())}))


def predict(model, w, ws, k, fallback):
    m = model.get(w)
    if not m:
        return fallback
    n = sum(m["prior"].values())
    best, bs = fallback, -1e18
    for y, py in m["prior"].items():
        c = m["cnt"][y]
        tot = sum(c.values())
        s = math.log(py / n) + sum(math.log((c.get(f, 0) + 0.5) / (tot + 0.5 * m["V"])) for f in feats(ws, k))
        if s > bs:
            best, bs = y, s
    return best


def evaluate(model_path, map_path, split):
    model, fmap = json.load(open(model_path)), std_map.load_map(map_path)
    ids = set(open(os.path.join(std_map.SPLITS, f"dial.{split}.txt")).read().split())
    tot = ok_nb = ok_maj = 0
    per = collections.defaultdict(lambda: [0, 0, 0])
    for g in std_map.gold_rows():
        if g["id"] not in ids:
            continue
        d, s = g["dial"].split(), g["std"].split()
        for k, w in enumerate(d):
            e = fmap.get(w)
            if e and e["decision"] == "ctx":
                p = predict(model, w, d, k, e["std"])
                tot += 1; ok_nb += p == s[k]; ok_maj += e["std"] == s[k]
                per[w][0] += 1; per[w][1] += p == s[k]; per[w][2] += e["std"] == s[k]
    top = sorted(per.items(), key=lambda x: -x[1][0])[:10]
    print(json.dumps({"split": split, "ctx_words": tot, "majority_acc": round(100 * ok_maj / tot, 1),
                      "classifier_acc": round(100 * ok_nb / tot, 1),
                      "per_form": {w: f"n={a} nb={100*b/a:.0f}% maj={100*c/a:.0f}%" for w, (a, b, c) in top}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        train(a[0], a[1], a[a.index("--exclude-split") + 1] if "--exclude-split" in a else None)
    elif cmd == "eval":
        evaluate(a[0], a[1], a[2])
