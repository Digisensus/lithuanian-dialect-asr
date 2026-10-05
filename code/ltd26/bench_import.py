import csv, hashlib, io, json, os, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
BENCH = STUDY.parent / "benchmarks"
FLEURS = BENCH / "fleurs_lt" / "lt_lt" / "test" / "0000.parquet"
CV_TAR = BENCH / "cv19_lt" / "audio" / "lt" / "test" / "lt_test_0.tar"
CV_TSV = BENCH / "cv19_lt" / "transcript" / "lt" / "test.tsv"


def _to_flac(args):
    src_bytes, dst = args
    import numpy as np, soundfile as sf
    try:
        import soxr
    except ImportError:
        soxr = None
    data, sr = sf.read(io.BytesIO(src_bytes), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        if soxr is not None:
            data = soxr.resample(data, sr, 16000)
        else:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.asarray(data, dtype="float32"), 16000, format="FLAC", subtype="PCM_16")
    h = hashlib.sha256(Path(dst).read_bytes()).hexdigest()
    return len(data), h


def fleurs_items():
    import pyarrow.parquet as pq
    t = pq.read_table(FLEURS).to_pylist()
    for k, r in enumerate(t):
        yield (f"fleurs-lt-{k:04d}", r["audio"]["bytes"], r["raw_transcription"], f"fleurs-lt-{k:04d}")


def cv_items():
    rows = {r["path"]: r for r in csv.DictReader(open(CV_TSV, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
    with tarfile.open(CV_TAR) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if not m.isfile() or name not in rows:
                continue
            r = rows[name]
            spk = "cv-" + hashlib.sha1(r["client_id"].encode()).hexdigest()[:10]
            yield (f"cv19-lt-{name.rsplit('_', 1)[-1].split('.')[0]}", tf.extractfile(m).read(), r["sentence"], spk)


def main():
    from whisperfleet.hub.config import HubConfig
    from whisperfleet.hub.store import AudioVersion, Database, Segment, SourceFile, TranscriptVersion
    cfg = HubConfig()
    db = Database(cfg.db_url)
    db.init()
    root = Path(cfg.dataset_root)
    manifest = []
    for tag, split, items in (("bench-fleurs-lt", "fleurs.lt.test", fleurs_items),
                              ("bench-cv19-lt", "cv.lt.test", cv_items)):
        sf_id = "sf-" + hashlib.sha256(tag.encode()).hexdigest()[:12]
        with db.session() as s:
            if s.get(SourceFile, sf_id) is not None:
                print(f"{tag}: already imported — skipped", flush=True)
                continue
        recs = list(items())
        dsts = [root / "audio" / "bench" / tag / f"{cid}.flac" for cid, *_ in recs]
        with ProcessPoolExecutor(os.cpu_count()) as pool:
            res = list(pool.map(_to_flac, [(b, str(d)) for (_c, b, _t, _s), d in zip(recs, dsts)], chunksize=16))
        with db.session() as s:
            s.add(SourceFile(id=sf_id, sha256=None, original_name=split, source_tag=tag, raw_path=str(BENCH),
                             duration_ms=0.0, channels=1, channel_mode="mono", status="processing",
                             language=cfg.google_language))
            s.flush()
            tot = 0.0
            for i, ((cid, _b, text, _spk), d, (n, sha)) in enumerate(zip(recs, dsts, res), start=1):
                dur = n * 1000.0 / 16000
                seg = Segment(source_file_id=sf_id, channel_tag="mono", seq_index=i, start_ms=0.0, end_ms=dur,
                              duration_ms=dur, review_status="reviewed", reviewed_by="import",
                              reviewed_at=time.time())
                s.add(seg)
                s.flush()
                s.add(AudioVersion(segment_id=seg.id, version_no=0, audio_path=str(d.relative_to(root)),
                                   duration_ms=dur, num_samples=n, sample_rate=16000, audio_sha256=sha,
                                   is_current=True))
                s.add(TranscriptVersion(segment_id=seg.id, version_no=0, text=text, engine="benchmark-original",
                                        is_current=True, created_by="import:ltd26"))
                tot += dur
            sf = s.get(SourceFile, sf_id)
            sf.duration_ms, sf.status = tot, "processed"
            s.add(sf)
            s.commit()
        ids = [cid for cid, *_ in recs]
        (STUDY / "splits" / "v1" / f"{split}.txt").write_text("\n".join(ids) + "\n")
        with open(STUDY / "layers" / f"{split}.v1.jsonl", "w", encoding="utf-8") as f:
            for cid, _b, text, spk in recs:
                f.write(json.dumps({"id": cid, "text": text, "speaker": spk}, ensure_ascii=False) + "\n")
        print(json.dumps({"source": tag, "split": split, "clips": len(ids), "hours": round(tot / 3.6e6, 2)}), flush=True)
        manifest.append(f"{split}.txt")
    if manifest:
        sp = STUDY / "splits" / "v1"
        with open(sp / "MANIFEST.bench.sha256", "a") as f:
            for name in manifest:
                f.write(f"{hashlib.sha256((sp / name).read_bytes()).hexdigest()}  {name}\n")


if __name__ == "__main__":
    main()
