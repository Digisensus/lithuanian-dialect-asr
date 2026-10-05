import collections, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SPLITS = os.path.join(ROOT, "splits", "v1")
LIEPA = "../liepa3"
DIAL = "../lithuanian-dialect-speech-liepa-3-100h-punctuated/data"
PHONE = "../lithuanian-phone-speech-liepa-3-429h-punctuated/data"
SOURCES = {"D": "dictaphone", "R": "radio", "T": "tv"}
AGE = {"3": "18-60", "4": "60+"}
GENDER = {"F": "female", "M": "male"}
EDGES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30.001]


def length_bin(d):
    return int(np.searchsorted(EDGES, d, side="right")) - 1


def durations(paths):
    import soundfile as sf
    out = []
    for p in paths:
        try:
            out.append(sf.info(os.path.join(LIEPA, p)).duration)
        except Exception:
            out.append(-1.0)
    return out


def dial_target():
    import pyarrow.parquet as pq
    train = set(open(os.path.join(SPLITS, "dial.train.txt")).read().split())
    hours = collections.Counter()
    for f in glob.glob(os.path.join(DIAL, "*.parquet")):
        t = pq.read_table(f, columns=["id", "gender", "age_group", "duration"]).to_pydict()
        for i, g, a, d in zip(t["id"], t["gender"], t["age_group"], t["duration"]):
            if i in train:
                hours[(g, a, length_bin(d))] += d / 3600
    return hours


def phone_heldout_texts():
    import pyarrow.parquet as pq
    held = set()
    for f in glob.glob(os.path.join(PHONE, "test-*.parquet")) + glob.glob(os.path.join(PHONE, "validation-*.parquet")):
        for t in pq.read_table(f, columns=["text_normalized"]).column(0).to_pylist():
            t = " ".join(t.split())
            if len(t.split()) >= 6:
                held.add(t)
    return held


def main():
    a = sys.argv[1:]
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 2026
    target = dial_target()
    held = phone_heldout_texts()
    cands = collections.defaultdict(list)
    skipped = collections.Counter()
    for line in open(os.path.join(LIEPA, "metadata.csv"), encoding="utf-8"):
        path, _, text = line.rstrip("\n").partition("|")
        parts = path.split("/")
        if len(parts) < 5 or parts[1] != "spon" or parts[2] != "S" or parts[3] not in SOURCES:
            continue
        code = parts[4]
        if code[1] not in AGE:
            continue
        text = " ".join(text.split())
        if not text:
            skipped["empty_text"] += 1
            continue
        if text in held:
            skipped["repeats_phone_test_dev_transcript"] += 1
            continue
        cands[(GENDER[code[0]], AGE[code[1]])].append((path, text, SOURCES[parts[3]]))
    rng = np.random.default_rng(seed)
    left = dict(target)
    chosen, shortfall = [], {}
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        for ga in sorted(cands):
            pool_ga = cands[ga]
            order = rng.permutation(len(pool_ga))
            spare = []
            pos = 0
            while pos < len(order) and any(left.get((*ga, b), 0) > 0 for b in range(len(EDGES) - 1)):
                block = [pool_ga[k] for k in order[pos:pos + 16000]]
                pos += 16000
                chunks = [[p for p, _, _ in block[c:c + 500]] for c in range(0, len(block), 500)]
                durs = [d for part in pool.map(durations, chunks) for d in part]
                for (p, t, s), d in zip(block, durs):
                    if not 0.5 <= d <= 30:
                        continue
                    key = (*ga, length_bin(d))
                    if left.get(key, 0) > 0:
                        chosen.append((p, t, s, d, *ga))
                        left[key] -= d / 3600
                    else:
                        spare.append((p, t, s, d, *ga))
            for b in range(len(EDGES) - 1):
                key = (*ga, b)
                if left.get(key, 0) > 0:
                    shortfall[f"{ga[0]}/{ga[1]}/{EDGES[b]}-{EDGES[b + 1]}s"] = round(left[key], 3)
                    spare.sort(key=lambda r: abs(length_bin(r[3]) - b))
                    while left[key] > 0 and spare:
                        r = spare.pop(0)
                        chosen.append(r)
                        left[key] -= r[3] / 3600
            print(json.dumps({"bucket": ga, "candidates": len(pool_ga), "headers_read": min(pos, len(order)),
                              "chosen_h": round(sum(r[3] for r in chosen if r[4:] == ga) / 3600, 2)}), flush=True)
    chosen.sort()
    ids = [os.path.basename(p)[:-5] for p, *_ in chosen]
    assert len(set(ids)) == len(ids)
    with open(os.path.join(SPLITS, "xspon.train.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(os.path.join(SPLITS, "xspon.train.meta.tsv"), "w") as f:
        f.write("id\tpath\tduration\tgender\tage_group\tsource\n")
        for i, (p, t, s, d, g, ag) in zip(ids, chosen):
            f.write(f"{i}\t{p}\t{d:.3f}\t{g}\t{ag}\t{s}\n")
    os.makedirs(os.path.join(ROOT, "layers"), exist_ok=True)
    with open(os.path.join(ROOT, "layers", "xspon.spoken.v1.jsonl"), "w", encoding="utf-8") as f:
        for i, (p, t, *_) in zip(ids, chosen):
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    hours_by = collections.defaultdict(float)
    src_by = collections.defaultdict(float)
    for p, t, s, d, g, ag in chosen:
        hours_by[f"{g}/{ag}"] += d / 3600
        src_by[s] += d / 3600
    stats = {"seed": seed, "clips": len(ids), "hours": round(sum(r[3] for r in chosen) / 3600, 2),
             "target_hours": round(sum(target.values()), 2),
             "hours_by_gender_age": {k: round(v, 2) for k, v in sorted(hours_by.items())},
             "target_by_gender_age": {f"{g}/{a}": round(sum(v for (gg, aa, _), v in target.items()
                                                            if (gg, aa) == (g, a)), 2)
                                      for g, a in sorted({k[:2] for k in target})},
             "hours_by_source": {k: round(v, 2) for k, v in sorted(src_by.items())},
             "cell_shortfall_filled_from_nearest_bin_h": shortfall,
             "candidates_skipped": dict(skipped)}
    json.dump(stats, open(os.path.join(SPLITS, "xspon_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
