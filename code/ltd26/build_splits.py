import argparse, collections, csv, glob, json, os, random

import pyarrow.parquet as pq

BLOCK = {"test": 25 * 60, "dev": 20 * 60, "seen": 5 * 60}
N = {"test": 3, "dev": 2, "seen": 3}
DOSES = (12, 25, 50)


def load(dataset_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(dataset_dir, "data", "*.parquet"))):
        rows += pq.read_table(f, columns=["id", "speaker_id", "region", "gender", "age_group", "duration"]).to_pylist()
    rows.sort(key=lambda r: r["id"])
    by = collections.defaultdict(list)
    for r in rows:
        by[r["speaker_id"]].append(r)
    return by


def pick(cands, n, rnd):
    cands = cands[:]
    rnd.shuffle(cands)
    chosen = []
    for need in (lambda c: c["gender"] == "male", lambda c: c["gender"] == "female",
                 lambda c: c["age"] == "18-60", lambda c: c["age"] == "60+"):
        if len(chosen) < n and not any(need(c) for c in chosen):
            hit = next((c for c in cands if need(c) and c not in chosen), None)
            if hit:
                chosen.append(hit)
    for c in cands:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def block(clips, seconds, rnd):
    total = sum(c["duration"] for c in clips)
    if total <= seconds:
        return clips[:]
    cum, starts = 0.0, []
    for i, c in enumerate(clips):
        if total - cum >= seconds:
            starts.append(i)
        cum += c["duration"]
    i = rnd.choice(starts)
    out, s = [], 0.0
    while i < len(clips) and s < seconds:
        out.append(clips[i]); s += clips[i]["duration"]; i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    by = load(a.dataset_dir)
    spk = {k: {"id": k, "region": v[0]["region"], "gender": v[0]["gender"], "age": v[0]["age_group"],
               "sec": sum(c["duration"] for c in v)} for k, v in by.items()}
    parts = collections.defaultdict(list)
    role = {}
    for reg in sorted({s["region"] for s in spk.values()}):
        pool = sorted((s for s in spk.values() if s["region"] == reg and s["sec"] >= 1800), key=lambda s: s["id"])
        for split in ("test", "dev"):
            for s in pick([c for c in pool if c["id"] not in role], N[split], rnd):
                role[s["id"]] = split
    for k, clips in sorted(by.items()):
        if k in role:
            b = block(clips, BLOCK[role[k]], rnd)
            ids = {c["id"] for c in b}
            parts["dial." + role[k]] += b
            parts["dial.excluded"] += [c for c in clips if c["id"] not in ids]
    train_spk = sorted(k for k in by if k not in role)
    for reg in sorted({spk[k]["region"] for k in train_spk}):
        pool = [spk[k] for k in train_spk if spk[k]["region"] == reg and spk[k]["sec"] >= 1200]
        for s in pick(pool, N["seen"], rnd):
            role[s["id"]] = "seen"
    for k in train_spk:
        clips = by[k]
        if role.get(k) == "seen":
            b = block(clips, BLOCK["seen"], rnd)
            ids = {c["id"] for c in b}
            parts["dial.seen"] += b
            parts["dial.train"] += [c for c in clips if c["id"] not in ids]
        else:
            parts["dial.train"] += clips
    queues = {}
    for k in train_spk:
        queues.setdefault(spk[k]["region"], []).append(k)
    for q in queues.values():
        rnd.shuffle(q)
    order, regs = [], sorted(queues)
    while any(queues.values()):
        for r in regs:
            if queues[r]:
                order.append(queues[r].pop(0))
    train_ids = {c["id"] for c in parts["dial.train"]}
    cum, dose_rows = 0.0, []
    for k in order:
        cum += sum(c["duration"] for c in by[k] if c["id"] in train_ids) / 3600
        dose_rows.append((k, round(cum, 2)))
    os.makedirs(a.out_dir, exist_ok=True)
    stats = {"seed": a.seed}
    for name, clips in parts.items():
        clips.sort(key=lambda c: c["id"])
        with open(os.path.join(a.out_dir, name + ".txt"), "w") as f:
            f.write("\n".join(c["id"] for c in clips) + "\n")
        stats[name] = {"clips": len(clips), "hours": round(sum(c["duration"] for c in clips) / 3600, 2),
                       "speakers": len({c["speaker_id"] for c in clips})}
    with open(os.path.join(a.out_dir, "dose_order.txt"), "w") as f:
        f.write("\n".join(f"{k}\t{h}" for k, h in dose_rows) + "\n")
    doses = {}
    for d in DOSES:
        n = next(i for i, (_, h) in enumerate(dose_rows, 1) if h >= d)
        doses[f"D{d}"] = {"speakers": n, "hours": dose_rows[n - 1][1]}
    doses["D80"] = {"speakers": len(dose_rows), "hours": dose_rows[-1][1]}
    stats["doses"] = doses
    with open(os.path.join(a.out_dir, "speakers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "region", "gender", "age_group", "minutes", "role"])
        for k in sorted(spk):
            s = spk[k]
            w.writerow([k, s["region"], s["gender"], s["age"], round(s["sec"] / 60, 1), role.get(k, "train")])
    with open(os.path.join(a.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))
    for split in ("test", "dev", "seen"):
        print(split, sorted((s, spk[s]["region"][0], spk[s]["gender"][0], spk[s]["age"]) for s, r in role.items() if r == split))


if __name__ == "__main__":
    main()
