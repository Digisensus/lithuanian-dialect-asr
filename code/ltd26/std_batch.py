import glob, json, os, sqlite3, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
W = os.path.join(ROOT, "work", "std")
DB = os.path.join(ROOT, "data", "std_lexicon.sqlite")
CHUNK, CHUNKS = 270, 8
MAX_CLIPS, MAX_NS = 250, 450


def plan(splits_dir, dataset_dir):
    import pyarrow.parquet as pq
    rows = {}
    for f in sorted(glob.glob(os.path.join(dataset_dir, "data", "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "speaker_id", "text_normalized"]).to_pylist():
            rows[r["id"]] = r
    ids = lambda name: [l for l in open(os.path.join(splits_dir, f"dial.{name}.txt")).read().split()]
    queue = [(i, "test") for i in ids("test")] + [(i, "dev") for i in ids("dev")] + [(i, "seen") for i in ids("seen")]
    train = set(ids("train"))
    by_spk = {}
    for i in sorted(train):
        by_spk.setdefault(rows[i]["speaker_id"], []).append(i)
    for line in open(os.path.join(splits_dir, "dose_order.txt")):
        spk = line.split("\t")[0]
        queue += [(i, "train:" + spk) for i in by_spk[spk]]
    assert len(queue) == len({i for i, _ in queue})
    os.makedirs(W, exist_ok=True)
    with open(os.path.join(W, "queue.tsv"), "w") as f:
        for n, (i, part) in enumerate(queue):
            f.write(f"{n // (CHUNK * CHUNKS) + 1}\t{i}\t{part}\n")
    print(json.dumps({"clips": len(queue), "batches": -(-len(queue) // (CHUNK * CHUNKS))}))


def prep(n):
    import pyarrow.parquet as pq
    q = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "queue.tsv"))]
    want = [i for b, i, _ in q if int(b) == n]
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "..", "lithuanian-dialect-speech-liepa-3-100h-punctuated", "data", "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "speaker_id", "text_normalized"]).to_pylist():
            rows[r["id"]] = r
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    read = {}

    def ns(words):
        out = []
        for k, w in enumerate(words):
            if w not in read:
                x = con.execute("select read from lex where word = ?", (w,)).fetchone()
                read[w] = x[0] if x else 0
            if read[w] < 3:
                out.append(k)
        return out

    d = os.path.join(W, "batches", f"b{n:03d}")
    os.makedirs(os.path.join(d, "in"), exist_ok=True)
    os.makedirs(os.path.join(d, "out"), exist_ok=True)
    recs, chunks, cur, cur_ns = [], [], [], 0
    for i in want:
        words = " ".join(rows[i]["text_normalized"].split()).split()
        recs.append({"id": i, "spk": rows[i]["speaker_id"], "dial": " ".join(words), "ns": ns(words)})
    for r in recs:
        if cur and (len(cur) >= MAX_CLIPS or cur_ns + len(r["ns"]) > MAX_NS):
            chunks.append(cur); cur, cur_ns = [], 0
        cur.append(r); cur_ns += len(r["ns"])
    if cur:
        chunks.append(cur)
    for k, ch in enumerate(chunks):
        with open(os.path.join(d, "in", f"chunk_{k:03d}.jsonl"), "w", encoding="utf-8") as f:
            for r in ch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size = max(len(c) for c in chunks)
    parts = sorted({p for b, _, p in q if int(b) == n})
    print(json.dumps({"batch": n, "clips": len(want), "chunks": len(chunks), "clips_per_chunk": [len(c) for c in chunks],
                      "ns_per_chunk": [sum(len(r["ns"]) for r in c) for c in chunks], "parts": parts[:6] + (["…"] if len(parts) > 6 else [])}))


def replan(chunk):
    q = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "queue.tsv"))]
    started = {int(os.path.basename(d)[1:]) for d in glob.glob(os.path.join(W, "batches", "b*"))}
    keep = [r for r in q if int(r[0]) in started]
    rest = [r for r in q if int(r[0]) not in started]
    first = max(started, default=0) + 1
    per = chunk * CHUNKS
    with open(os.path.join(W, "queue.tsv"), "w") as f:
        for r in keep:
            f.write("\t".join(r) + "\n")
        for k, (_, i, part) in enumerate(rest):
            f.write(f"{first + k // per}\t{i}\t{part}\n")
    print(json.dumps({"kept_batches": sorted(started), "next_batch": first, "clips_per_batch": per,
                      "remaining_batches": -(-len(rest) // per), "remaining_clips": len(rest)}))


def finish(n):
    import collections, subprocess
    d = os.path.join(W, "batches", f"b{n:03d}")
    code = os.path.join(ROOT, "code", "ltd26")
    pol = subprocess.run([sys.executable, os.path.join(code, "std_policy.py"), d], capture_output=True, text=True).stdout.strip()
    subprocess.run([sys.executable, os.path.join(code, "std_validate.py"), d, "--merge"], capture_output=True, text=True)
    r = json.load(open(os.path.join(d, "report.json")))
    rows = [json.loads(l) for l in open(os.path.join(d, "std.jsonl"), encoding="utf-8")]
    m = collections.defaultdict(collections.Counter)
    for g in rows:
        for a, b in zip(g["dial"].split(), g["std"].split()):
            m[a][b] += 1
    incons = sorted(((a, c) for a, c in m.items() if len(c) > 1 and sum(c.values()) >= 4), key=lambda x: -sum(x[1].values()))
    oversights = [(a, dict(c)) for a, c in incons if c.get(a) and c.most_common(1)[0][0] != a and c[a] <= 0.25 * sum(c.values())]
    led = [l.split("\t") for l in open(os.path.join(W, "ledger.tsv"))]
    tok = sum(int(x[1]) for x in led if int(x[0]) == n)
    inputs = len(glob.glob(os.path.join(d, "in", "chunk_*.jsonl")))
    print(json.dumps({"batch": n, "tokens": tok, "agents_logged": sum(1 for x in led if int(x[0]) == n), "chunks": inputs,
                      "clips_ok": r["clips_ok"], "errors": r["errors"], "error_examples": r["error_examples"][:5],
                      "changed_pct": r["changed_pct"], "unsure_pct": r["unsure_words_pct"], "warnings": r.get("warnings"),
                      "policy": json.loads(pol) if pol else None, "inconsistent_forms": len(incons),
                      "likely_oversights": oversights[:10],
                      "merges": sum(1 for g in rows for w in g["std"].split() if "+" in w)}, ensure_ascii=False))


def replan_chunks(chunks_per_batch=CHUNKS):
    q = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "queue.tsv"))]
    started = {int(os.path.basename(d)[1:]) for d in glob.glob(os.path.join(W, "batches", "b*"))}
    keep = [r for r in q if int(r[0]) in started]
    rest = [r for r in q if int(r[0]) not in started]
    import pyarrow.parquet as pq
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "..", "lithuanian-dialect-speech-liepa-3-100h-punctuated", "data", "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "text_normalized"]).to_pylist():
            rows[r["id"]] = r["text_normalized"].split()
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    read = {}

    def n_ns(words):
        k = 0
        for w in words:
            if w not in read:
                x = con.execute("select read from lex where word = ?", (w,)).fetchone()
                read[w] = x[0] if x else 0
            k += read[w] < 3
        return k

    batch, chunk_i, clips, ns_ = max(started, default=0) + 1, 0, 0, 0
    out = []
    for _, i, part in rest:
        k = n_ns(rows[i])
        if clips and (clips >= MAX_CLIPS or ns_ + k > MAX_NS):
            chunk_i, clips, ns_ = chunk_i + 1, 0, 0
            if chunk_i == chunks_per_batch:
                batch, chunk_i = batch + 1, 0
        out.append((batch, i, part))
        clips, ns_ = clips + 1, ns_ + k
    with open(os.path.join(W, "queue.tsv"), "w") as f:
        for r in keep:
            f.write("\t".join(r) + "\n")
        for b, i, part in out:
            f.write(f"{b}\t{i}\t{part}\n")
    nb = out[-1][0] - (max(started, default=0)) if out else 0
    print(json.dumps({"kept_batches": sorted(started), "remaining_batches": nb, "remaining_clips": len(out),
                      "avg_clips_per_batch": round(len(out) / max(1, nb))}))


def requeue_weak():
    layer = os.path.join(ROOT, "layers", "dial.standard.v1.jsonl")
    weak = set()
    for l in open(layer, encoding="utf-8"):
        r = json.loads(l)
        if r["tier"] == "silver" and r["split"] == "train" and r["unsure"]:
            weak.add(r["id"])
    q = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "queue.tsv"))]
    started = {int(os.path.basename(d)[1:]) for d in glob.glob(os.path.join(W, "batches", "b*"))}
    keep = [r for r in q if int(r[0]) in started]
    rest = [r for r in q if int(r[0]) not in started and r[1] in weak]
    assert len(rest) == len(weak - {r[1] for r in keep}), "weak clips missing from the queue"
    with open(os.path.join(W, "queue.tsv"), "w") as f:
        for r in keep + [[str(max(started) + 1), i, p] for _, i, p in rest]:
            f.write("\t".join(r) + "\n")
    replan_chunks()


def join(n, chunk):
    d = os.path.join(W, "batches", f"b{n:03d}", "out")
    parts = sorted(glob.glob(os.path.join(d, "parts", f"{chunk}.p[0-9][0-9].jsonl")))
    lines = [l if l.endswith("\n") else l + "\n" for p in parts for l in open(p, encoding="utf-8") if l.strip()]
    with open(os.path.join(d, f"{chunk}.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(json.dumps({"parts": [os.path.basename(p) for p in parts], "lines": len(lines)}))


def tokens(n, values):
    with open(os.path.join(W, "ledger.tsv"), "a") as f:
        for v in values:
            f.write(f"{n}\t{int(v)}\t{time.strftime('%Y-%m-%d %H:%M')}\n")


def status():
    q = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "queue.tsv"))]
    led = [l.split("\t") for l in open(os.path.join(W, "ledger.tsv"))] if os.path.exists(os.path.join(W, "ledger.tsv")) else []
    per = {}
    for x in led:
        per.setdefault(int(x[0]), [0, 0])
        per[int(x[0])][0] += int(x[1])
        per[int(x[0])][1] += 1
    chunks = {}
    for d in sorted(glob.glob(os.path.join(W, "batches", "b*"))):
        n = int(os.path.basename(d)[1:])
        chunks[n] = (len(glob.glob(os.path.join(d, "out", "chunk_*.jsonl"))), len(glob.glob(os.path.join(d, "in", "chunk_*.jsonl"))))
    print(json.dumps({"batches": {n: {"chunks_written": f"{c[0]}/{c[1]}", "agents_logged": per.get(n, [0, 0])[1],
                                      "tokens": per.get(n, [0, 0])[0]} for n, c in chunks.items()},
                      "tokens_total": sum(v[0] for v in per.values()), "clips_total": len(q)}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan(sys.argv[2], sys.argv[3])
    elif cmd == "prep":
        prep(int(sys.argv[2]))
    elif cmd == "replan":
        replan(int(sys.argv[2]))
    elif cmd == "replan-chunks":
        replan_chunks()
    elif cmd == "finish":
        finish(int(sys.argv[2]))
    elif cmd == "requeue-weak":
        requeue_weak()
    elif cmd == "join":
        join(int(sys.argv[2]), sys.argv[3])
    elif cmd == "tokens":
        tokens(int(sys.argv[2]), sys.argv[3:])
    elif cmd == "status":
        status()
