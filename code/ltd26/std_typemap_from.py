import collections, json, sys


def main():
    types_f, std_f, out_f = sys.argv[1:4]
    m = collections.defaultdict(collections.Counter)
    for l in open(std_f, encoding="utf-8"):
        g = json.loads(l)
        for d, s in zip(g["dial"].split(), g["std"].split()):
            m[d][s] += 1
    n = collections.Counter()
    with open(out_f, "w", encoding="utf-8") as f:
        for l in open(types_f, encoding="utf-8"):
            t = json.loads(l)["type"]
            c = m.get(t)
            if not c:
                continue
            forms = [s for s, _ in c.most_common()]
            if len(forms) == 1:
                d = {"type": t, "decision": "keep" if forms[0] == t else "std", "std": forms[0]}
            else:
                d = {"type": t, "decision": "ctx", "cands": forms}
            n[d["decision"]] += 1
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(dict(n)))


if __name__ == "__main__":
    main()
