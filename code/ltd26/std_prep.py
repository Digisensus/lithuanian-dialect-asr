import argparse, glob, json, os, unicodedata

import pyarrow.parquet as pq


def load(dataset_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(dataset_dir, "data", "*.parquet"))):
        rows += pq.read_table(f, columns=["id", "speaker_id", "text_normalized"]).to_pylist()
    rows.sort(key=lambda r: r["id"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ids")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--chunk", type=int, default=265)
    a = ap.parse_args()
    rows = load(a.dataset_dir)
    if not a.all:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        rows = [r for r in rows if r["id"] in keep]
        assert len(rows) == len(keep), (len(rows), len(keep))
    os.makedirs(os.path.join(a.out_dir, "in"), exist_ok=True)
    chars = set()
    n = 0
    for i in range(0, len(rows), a.chunk):
        with open(os.path.join(a.out_dir, "in", f"chunk_{i // a.chunk:03d}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows[i:i + a.chunk]:
                dial = " ".join(unicodedata.normalize("NFC", r["text_normalized"]).split())
                chars |= set(dial)
                f.write(json.dumps({"id": r["id"], "spk": r["speaker_id"], "dial": dial}, ensure_ascii=False) + "\n")
        n += 1
    print({"clips": len(rows), "chunks": n, "chars": "".join(sorted(chars))})


if __name__ == "__main__":
    main()
