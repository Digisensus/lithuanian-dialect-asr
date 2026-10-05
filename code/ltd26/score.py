import collections, json, os, random, re, sys, unicodedata

VERSION = "score.py 1.3 (2026-10-04)"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalise(text):
    t = unicodedata.normalize("NFC", text or "").lower()
    t = PUNCT.sub(" ", t)
    return " ".join(t.split())


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            s, d, a = prev[j - 1], prev[j], cur[j - 1]
            cur[j] = min((s[0] + 1, s[1] + 1, s[2], s[3]), (d[0] + 1, d[1], d[2] + 1, d[3]), (a[0] + 1, a[1], a[2], a[3] + 1))
        prev = cur
    return prev[m]


def either_align(slots, hyp):
    n, m = len(slots), len(hyp)
    INF = 10 ** 9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0] = list(range(m + 1))
    for i in range(1, n + 1):
        alts = slots[i - 1]
        maxlen = max(len(a) for a in alts)
        for j in range(m + 1):
            best, arg = INF, None
            limit = min(j, 2) if maxlen == 1 else j
            for k in range(limit + 1):
                if dp[i - 1][j - k] >= INF:
                    continue
                seg = hyp[j - k:j]
                c = min(edit_distance(list(a), seg)[0] for a in alts)
                if dp[i - 1][j - k] + c < best:
                    best, arg = dp[i - 1][j - k] + c, (i - 1, j - k, c)
            if j and dp[i][j - 1] + 1 < best:
                best, arg = dp[i][j - 1] + 1, (i, j - 1, 1)
            dp[i][j] = best
            back[i][j] = arg
    j_end = min(range(m + 1), key=lambda j: dp[n][j] + (m - j))
    total = dp[n][j_end] + (m - j_end)
    per, i, j = [0] * n, n, j_end
    while i:
        pi, pj, c = back[i][j]
        per[i - 1] += c
        i, j = pi, pj
    if n:
        per[0] += j
        per[-1] += m - j_end
    return total, per


def load_layer(path):
    out = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if "dial" not in r:
            r = {**r, "dial": r.get("text", ""), "std": r.get("text", "")}
        out[r["id"]] = r
    return out


def load_split(name):
    base = name.rsplit(".v", 1)[0]
    folder = os.path.join("splits", "v1")
    return open(os.path.join(ROOT, folder, f"{base}.txt")).read().split()


def clip_scores(ref_row, hyp_text):
    hyp = normalise(hyp_text).split()
    dial = normalise(ref_row["dial"]).split()
    std = normalise(ref_row["std"]).split()
    al = ref_row.get("alignment") or [[d, d, "same", ""] for d in dial]
    slots = [(tuple(normalise(a[0]).split()), tuple(normalise(a[1].replace("+", " ")).split())) for a in al]
    e_dial = edit_distance(dial, hyp)
    e_std = edit_distance(std, hyp)
    e_either, per = either_align(slots, hyp)
    tags = collections.Counter()
    tag_n = collections.Counter()
    for a, e in zip(al, per):
        tag_n[a[2]] += 1
        tags[a[2]] += e
    c_dial = edit_distance(list(" ".join(dial)), list(" ".join(hyp)))[0]
    return {"n": len(dial), "n_std": len(std), "e_dial": e_dial[0], "e_std": e_std[0], "e_either": e_either,
            "sub": e_dial[1], "del": e_dial[2], "ins": e_dial[3], "chars": len(" ".join(dial)), "e_char": c_dial,
            "tag_err": dict(tags), "tag_n": dict(tag_n)}


def aggregate(per_clip, ids):
    tot = collections.Counter()
    tag_e, tag_n = collections.Counter(), collections.Counter()
    spk = collections.defaultdict(collections.Counter)
    for i in ids:
        c = per_clip[i]
        for k in ("n", "n_std", "e_dial", "e_std", "e_either", "sub", "del", "ins", "chars", "e_char"):
            tot[k] += c[k]
        tag_e.update(c["tag_err"]); tag_n.update(c["tag_n"])
        sk = c.get("spk") or i
        spk[sk]["n"] += c["n"]; spk[sk]["e"] += c["e_dial"]
    pct = lambda a, b: round(100 * a / b, 3) if b else None
    region = collections.defaultdict(lambda: [0, 0])
    for s, c in spk.items():
        region[s[0]][0] += c["e"]; region[s[0]][1] += c["n"]
    return {"clips": len(ids), "words": tot["n"], "words_standard": tot["n_std"],
            "wer_dialect": pct(tot["e_dial"], tot["n"]), "wer_standard": pct(tot["e_std"], tot["n_std"]),
            "wer_either": pct(tot["e_either"], tot["n"]), "cer_dialect": pct(tot["e_char"], tot["chars"]),
            "sub": tot["sub"], "del": tot["del"], "ins": tot["ins"],
            "wer_by_region": {r: pct(e, n) for r, (e, n) in sorted(region.items())},
            "wer_mean_over_speakers": round(sum(100 * c["e"] / c["n"] for c in spk.values() if c["n"]) / max(1, len(spk)), 3),
            "either_err_by_tag": {t: pct(tag_e[t], tag_n[t]) for t in sorted(tag_n)}, "scorer": VERSION}


def bootstrap(per_clip, ids, key="e_either", other=None, n_boot=10000, seed=2026):
    by = collections.defaultdict(list)
    for i in ids:
        by[per_clip[i].get("spk") or i].append(i)
    spk = sorted(by)
    rnd = random.Random(seed)
    den = "n_std" if key == "e_std" else "n"
    tot = lambda pc: {s: (sum(pc[i][key] for i in by[s]), sum(pc[i][den] for i in by[s])) for s in spk}
    ta, tb = tot(per_clip), (tot(other) if other else None)
    def wer(sample, t):
        e = sum(t[s][0] for s in sample); n = sum(t[s][1] for s in sample)
        return 100 * e / n
    vals = []
    for _ in range(n_boot):
        sample = [rnd.choice(spk) for _ in spk]
        v = wer(sample, ta)
        vals.append(v - wer(sample, tb) if other else v)
    vals.sort()
    return [round(vals[int(0.025 * n_boot)], 3), round(vals[int(0.975 * n_boot)], 3)]


def score_file(hyps_path, split, layer_path):
    layer = load_layer(layer_path)
    ids = load_split(split)
    hyps = {h["id"]: h["hyp"] for h in map(json.loads, open(hyps_path, encoding="utf-8"))}
    missing = [i for i in ids if i not in hyps]
    per = {i: {**clip_scores(layer[i], hyps.get(i, "")),
               "spk": layer[i].get("speaker") or (i[:3] if split.startswith("dial.") else i)}
           for i in ids}
    res = aggregate(per, ids)
    res["missing_hyps"] = len(missing)
    return res, per, ids


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    layer = os.path.join(ROOT, opt("--layer", "layers/dial.standard.v2.jsonl"))
    if cmd == "hyps":
        res, per, ids = score_file(a[0], opt("--split"), layer)
        res["ci95_wer_either"] = bootstrap(per, ids, "e_either", n_boot=int(opt("--boot", 2000)))
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif cmd == "paired":
        key = {"dialect": "e_dial", "standard": "e_std", "either": "e_either"}[opt("--view", "either")]
        ra, pa, ids = score_file(a[0], opt("--split"), layer)
        rb, pb, _ = score_file(a[1], opt("--split"), layer)
        print(json.dumps({"view": opt("--view", "either"), "A": ra, "B": rb,
                          "diff_ci95_A_minus_B": bootstrap(pa, ids, key, other=pb, n_boot=int(opt("--boot", 10000)))},
                         ensure_ascii=False, indent=1))
