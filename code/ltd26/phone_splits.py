import collections, glob, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "..", "splits", "v1")
DEFAULT = "../lithuanian-phone-speech-liepa-3-429h-punctuated/data"
PART = {"test": "test", "validation": "dev", "train": "train"}


def scan(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    split = PART[os.path.basename(path).split("-")[0]]
    out = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["id", "audio", "text_normalized", "duration"]).to_pydict()
        for i, au, tx, du in zip(t["id"], t["audio"], t["text_normalized"], t["duration"]):
            out.append((i, split, hashlib.sha1(au["bytes"]).hexdigest(), " ".join(tx.split()), float(du)))
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        rows = [r for part in pool.map(scan, sorted(glob.glob(os.path.join(src, "*.parquet")))) for r in part]
    for i, split, *_ in rows:
        b = int(hashlib.sha1(i.encode()).hexdigest(), 16) % 100
        assert split == ("test" if b == 0 else "dev" if b == 1 else "train"), (i, split, b)
    held_audio = {h for _, s, h, _, _ in rows if s != "train"}
    held_text = collections.Counter(t for _, s, _, t, _ in rows if s != "train" and len(t.split()) >= 6)
    ids = collections.defaultdict(list)
    hours = collections.Counter()
    dup_audio = []
    for i, s, h, t, d in rows:
        if s == "train" and (h in held_audio or t in held_text):
            dup_audio.append(i)
            s = "excluded"
        ids[s].append(i)
        hours[s] += d / 3600
    audio_within = sum(v - 1 for v in collections.Counter(h for _, _, h, _, _ in rows).values() if v > 1)
    train_text = collections.Counter(t for _, s, _, t, _ in rows if s == "train")
    shared_text = {t for t in held_text if train_text[t]}
    for s, v in ids.items():
        with open(os.path.join(SPLITS, f"phone.{s}.txt"), "w") as f:
            f.write("\n".join(sorted(v)) + "\n")
    stats = {
        "source": "Digisensus/lithuanian-phone-speech-liepa-3-429h-punctuated (published split, sha1(id) mod 100)",
        "clips": {s: len(v) for s, v in ids.items()},
        "hours": {s: round(h, 2) for s, h in hours.items()},
        "train_moved_to_excluded_(same_audio_or_transcript_as_test_dev)": len(dup_audio),
        "identical_audio_pairs_all_splits": audio_within,
        "test_dev_clips_with_transcript_(>=6 words)_also_in_train": sum(1 for _, s, _, t, _ in rows
                                                                          if s != "train" and t in shared_text),
        "speaker_disjoint": False,
    }
    json.dump(stats, open(os.path.join(SPLITS, "phone_stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
