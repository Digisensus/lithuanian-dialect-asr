import glob, io, json, os, sys, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SR, MAX_SEC, MIN_SEC, GRID = 16000, 10.0, 0.5, 4000
META = ("gender", "age_group", "speaker_id", "duration")


def _crop(x, sec=MAX_SEC):
    n = int(min(sec, MAX_SEC) * SR)
    if len(x) > n:
        s = (len(x) - n) // 2
        x = x[s:s + n]
    if len(x) < int(MIN_SEC * SR):
        x = np.pad(x, (0, int(MIN_SEC * SR) - len(x)))
    n = len(x) // GRID * GRID
    s = (len(x) - n) // 2
    return np.ascontiguousarray(x[s:s + n], np.float32)


def read_row_group(args):
    path, rg = args
    import pyarrow.parquet as pq, soundfile as sf
    pf = pq.ParquetFile(path)
    cols = [c for c in ("id", "audio") + META if c in pf.schema_arrow.names]
    t = pf.read_row_group(rg, columns=cols).to_pydict()
    wavs = []
    for au in t["audio"]:
        x, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
        wavs.append(_crop(x.mean(axis=1) if x.ndim > 1 else x))
    return rg, t["id"], {k: t[k] for k in META if k in t}, wavs


def read_files(args):
    k, rows, tel = args
    import soundfile as sf, torch, torchaudio.functional as AF
    torch.set_num_threads(1)
    wavs = []
    for cid, path, spk, crop in rows:
        x, sr = sf.read(path, dtype="float32")
        x = torch.from_numpy(x.mean(axis=1) if x.ndim > 1 else x)
        if tel:
            x = AF.resample(AF.resample(x, sr, 8000), 8000, SR)
        elif sr != SR:
            x = AF.resample(x, sr, SR)
        wavs.append(_crop(x.numpy(), float(crop)))
    return k, [r[0] for r in rows], {"speaker_id": [r[2] for r in rows], "duration": [float(r[3]) for r in rows]}, wavs


class Embedder:

    def __init__(self, bs):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch, self.bs = torch, bs
        self.dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        here = os.path.dirname(os.path.abspath(__file__))
        self.enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                  savedir=os.path.join(here, "..", "..", "work", "spk", "ecapa_model"),
                                                  run_opts={"device": self.dev})
        self.enc.eval()
        self.pending = {}
        self.done = {}

    def add(self, key, wav):
        L = len(wav)
        self.pending.setdefault(L, []).append((key, wav))
        if len(self.pending[L]) >= self.bs:
            self._run(L, self.pending.pop(L))

    def flush(self):
        for L in list(self.pending):
            self._run(L, self.pending.pop(L))

    def _run(self, L, items):
        torch = self.torch
        nb = 1 << (len(items) - 1).bit_length()
        w = np.zeros((nb, L), np.float32)
        w[:len(items)] = np.stack([x for _, x in items])
        with torch.inference_mode():
            e = self.enc.encode_batch(torch.from_numpy(w).to(self.dev)).squeeze(1)
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16)
        for r, (k, _) in enumerate(items):
            self.done[k] = e[r]


def run_tasks(pool, fn, tasks, emb, inflight):
    q, it, results = deque(), iter(tasks), {}
    for t in it:
        q.append(pool.submit(fn, t))
        if len(q) >= inflight:
            break
    while q:
        k, ids, meta, wavs = q.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            q.append(pool.submit(fn, nxt))
        for j, x in enumerate(wavs):
            emb.add((k, j), x)
        results[k] = (ids, meta)
    emb.flush()
    ids, embs, meta = [], [], {}
    for k in sorted(results):
        rid, rmeta = results[k]
        ids += rid
        embs += [emb.done.pop((k, j)) for j in range(len(rid))]
        for m, v in rmeta.items():
            meta.setdefault(m, []).extend(v)
    return ids, np.stack(embs), meta


def main():
    a = sys.argv[1:]
    list_mode = a[0] == "--list"
    if list_mode:
        a = a[1:]
    src, out, a = a[0], a[1], a[2:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    pattern, workers, bs = opt("--pattern", "*.parquet"), opt("--workers", 15), opt("--batch", 128)
    os.makedirs(out, exist_ok=True)
    emb = Embedder(bs)
    t0, total = time.time(), 0

    def report(name, n):
        nonlocal total
        total += n
        el = time.time() - t0
        print(json.dumps({"part": name, "clips": n, "total": total, "clips_per_s": round(total / el, 1),
                          "elapsed_min": round(el / 60, 1)}), flush=True)

    with ProcessPoolExecutor(workers) as pool:
        if list_mode:
            rows = [l.rstrip("\n").split("\t") for l in open(src)][1:]
            tel = "--telephone" in a
            tasks = [(k, rows[c:c + 64], tel) for k, c in enumerate(range(0, len(rows), 64))]
            ids, e, meta = run_tasks(pool, read_files, tasks, emb, workers * 4)
            np.savez(os.path.join(out, "list.npz"), ids=np.array(ids), emb=e,
                     gender=np.array(["x"] * len(ids)), age_group=np.array(["all"] * len(ids)),
                     **{k: np.array(v) for k, v in meta.items()})
            report("list", len(ids))
            return
        import pyarrow.parquet as pq
        for sh in sorted(glob.glob(os.path.join(src, pattern))):
            dst = os.path.join(out, os.path.basename(sh).replace(".parquet", ".npz"))
            if os.path.exists(dst):
                continue
            tasks = [(sh, rg) for rg in range(pq.ParquetFile(sh).metadata.num_row_groups)]
            ids, e, meta = run_tasks(pool, read_row_group, tasks, emb, workers * 4)
            np.savez(dst + ".tmp.npz", ids=np.array(ids), emb=e, **{k: np.array(v) for k, v in meta.items()})
            os.replace(dst + ".tmp.npz", dst)
            report(os.path.basename(sh), len(ids))


if __name__ == "__main__":
    main()
