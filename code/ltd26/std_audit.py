import collections, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from std_batch import DB, MAX_CLIPS, MAX_NS

ROOT = os.path.join(HERE, "..", "..")
LAYER = os.path.join(ROOT, "layers", "dial.standard.v1.jsonl")


TIERS = ("silver",)


def silver_train():
    for l in open(LAYER, encoding="utf-8"):
        r = json.loads(l)
        if r["tier"] in TIERS and r["split"] == "train" and r["id"] not in EXCLUDE:
            yield r


EXCLUDE = set()


def sample(out, clips=1000, block=10, seed=2026):
    import sqlite3
    rows = list(silver_train())
    rnd = random.Random(seed)
    starts = [k for k in range(len(rows) - block + 1)
              if rows[k]["id"][:5] == rows[k + block - 1]["id"][:5]]
    rnd.shuffle(starts)
    taken, picked = set(), []
    for s in starts:
        if len(picked) * block >= clips:
            break
        if any(k in taken for k in range(s, s + block)):
            continue
        taken.update(range(s, s + block))
        picked.append(s)
    picked.sort()
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    read = {}

    def ns(words):
        o = []
        for k, w in enumerate(words):
            if w not in read:
                x = con.execute("select read from lex where word = ?", (w,)).fetchone()
                read[w] = x[0] if x else 0
            if read[w] < 3:
                o.append(k)
        return o

    recs = [{"id": rows[k]["id"], "spk": rows[k]["id"][:3], "dial": rows[k]["dial"], "ns": ns(rows[k]["dial"].split())}
            for s in picked for k in range(s, s + block)]
    chunks, cur, cur_ns = [], [], 0
    for r in recs:
        if cur and (len(cur) >= MAX_CLIPS or cur_ns + len(r["ns"]) > MAX_NS):
            chunks.append(cur); cur, cur_ns = [], 0
        cur.append(r); cur_ns += len(r["ns"])
    if cur:
        chunks.append(cur)
    os.makedirs(os.path.join(out, "in"), exist_ok=True)
    os.makedirs(os.path.join(out, "out"), exist_ok=True)
    for k, ch in enumerate(chunks):
        with open(os.path.join(out, "in", f"chunk_{k:03d}.jsonl"), "w", encoding="utf-8") as f:
            for r in ch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"clips": len(recs), "blocks": len(picked), "speakers": len({r["spk"] for r in recs}),
                      "regions": dict(collections.Counter(r["spk"][0] for r in recs)), "chunks": len(chunks),
                      "clips_per_chunk": [len(c) for c in chunks], "ns_per_chunk": [sum(len(r["ns"]) for r in c) for c in chunks]}))


def score(out, boot=2000, seed=2026):
    aud = {g["id"]: g for g in map(json.loads, open(os.path.join(out, "std.jsonl"), encoding="utf-8"))}
    tot, ok = collections.Counter(), collections.Counter()
    spk = collections.defaultdict(lambda: [0, 0])
    ch_tot = ch_ok = clips = exact = 0
    ex = collections.defaultdict(list)
    for r in silver_train():
        a = aud.get(r["id"])
        if not a:
            continue
        clips += 1
        ref = a["std"].split()
        al = r["alignment"]
        assert len(ref) == len(al), r["id"]
        exact += all(x[1] == y for x, y in zip(al, ref))
        for (d, s, _, src), y in zip(al, ref):
            hit = s == y
            tot[src] += 1; ok[src] += hit
            sp = spk[r["id"][:3]]; sp[0] += 1; sp[1] += hit
            if s != d or y != d:
                ch_tot += 1; ch_ok += hit
            if not hit and len(ex[src]) < 10:
                ex[src].append(f"{d}: silver {s} | audit {y}")
    words = sum(tot.values())
    acc = sum(ok.values()) / words
    rnd, keys, bs = random.Random(seed), list(spk), []
    for _ in range(boot):
        smp = [spk[rnd.choice(keys)] for _ in keys]
        bs.append(sum(x[1] for x in smp) / sum(x[0] for x in smp))
    bs.sort()
    res = {"clips": clips, "speakers": len(spk), "words": words, "word_accuracy_pct": round(100 * acc, 2),
           "ci95_speaker_bootstrap": [round(100 * bs[int(0.025 * boot)], 2), round(100 * bs[int(0.975 * boot)], 2)],
           "changed_words": ch_tot, "changed_word_accuracy_pct": round(100 * ch_ok / max(1, ch_tot), 2),
           "clip_exact_pct": round(100 * exact / max(1, clips), 1), "gate_G2d_93pct": acc >= 0.93,
           "by_source": {s: {"words": tot[s], "share_pct": round(100 * tot[s] / words, 1),
                             "accuracy_pct": round(100 * ok[s] / tot[s], 1)} for s in sorted(tot)},
           "examples": dict(ex)}
    json.dump(res, open(os.path.join(out, "audit_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    if "--layer" in a:
        LAYER = os.path.join(ROOT, a[a.index("--layer") + 1])
    if "--all-train" in a:
        TIERS = ("silver", "perclip", "gold")
    if "--exclude" in a:
        EXCLUDE.update(json.loads(l)["id"] for l in open(a[a.index("--exclude") + 1], encoding="utf-8"))
    if cmd == "sample":
        sample(a[0], opt("--clips", 1000), opt("--block", 10), opt("--seed", 2026))
    elif cmd == "score":
        score(a[0])
