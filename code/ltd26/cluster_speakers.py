import collections, glob, json, os, sys

import numpy as np


def load(emb_dir):
    parts = [np.load(f) for f in sorted(glob.glob(os.path.join(emb_dir, "*.npz")))]
    cat = lambda k: np.concatenate([p[k] for p in parts]) if k in parts[0].files else None
    return {k: cat(k) for k in ("ids", "emb", "gender", "age_group", "speaker_id", "duration")}


def bucket_key(g, a):
    return f"{str(g)[:1]}{str(a).split('-')[0].rstrip('+')}"


def micro_clusters(x, micro, seed=2026):
    import faiss
    k = max(2, len(x) // micro)
    km = faiss.Kmeans(x.shape[1], k, niter=25, spherical=True, seed=seed, verbose=False,
                      max_points_per_centroid=max(256, micro * 4))
    km.train(x)
    _, lab = km.index.search(x, 1)
    lab = lab[:, 0]
    cent = np.zeros((k, x.shape[1]), np.float32)
    np.add.at(cent, lab, x)
    used = np.unique(lab)
    cent = cent[used]
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    remap = np.full(k, -1)
    remap[used] = np.arange(len(used))
    return remap[lab], cent, np.bincount(remap[lab], minlength=len(used))


def linkage_of(cent):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    return linkage(pdist(cent.astype(np.float64), "cosine"), "average")


def cut(Z, t):
    from scipy.cluster.hierarchy import fcluster
    return fcluster(Z, t, "distance") - 1


def prepare(d, micro):
    keys = np.array([bucket_key(g, a) for g, a in zip(d["gender"], d["age_group"])])
    out = {}
    for b in sorted(set(keys)):
        idx = np.where(keys == b)[0]
        x = d["emb"][idx].astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lab, cent, _ = micro_clusters(x, micro)
        out[b] = (idx, lab, linkage_of(cent))
        print(json.dumps({"bucket": b, "clips": len(idx), "micro": len(cent)}), file=sys.stderr, flush=True)
    return out


def labels_at(prep, n, t):
    lab = np.empty(n, object)
    for b, (idx, micro, Z) in prep.items():
        top = cut(Z, t)
        lab[idx] = [f"{b}-{c:05d}" for c in top[micro]]
    return lab


def purity(clusters, speakers):
    by_c, by_s = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for c, s in zip(clusters, speakers):
        by_c[c][s] += 1
        by_s[s][c] += 1
    n = len(clusters)
    return (sum(v.most_common(1)[0][1] for v in by_c.values()) / n,
            sum(v.most_common(1)[0][1] for v in by_s.values()) / n, len(by_c))


def leakage(clusters, speakers, share=0.10, reps=20, seed=2026):
    rng = np.random.default_rng(seed)
    speakers = np.asarray(speakers)
    cl, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sp, sinv = np.unique(speakers, return_inverse=True)
    tot = np.bincount(sinv, minlength=len(sp))
    size = np.bincount(inv, minlength=len(cl))
    any_, mass = [], []
    for _ in range(reps):
        chosen = np.zeros(len(cl), bool)
        n = 0
        for c in rng.permutation(len(cl)):
            if n >= share * len(clusters):
                break
            chosen[c] = True
            n += size[c]
        t = chosen[inv]
        in_test = np.bincount(sinv[t], minlength=len(sp))
        out = tot - in_test
        any_.append(float((out[sinv[t]] > 0).mean()))
        mass.append(float((out[sinv[t]] / tot[sinv[t]]).mean()))
    return float(np.mean(any_)), float(np.mean(mass))


def main():
    cmd, emb_dir = sys.argv[1], sys.argv[2]
    a = sys.argv[3:]
    opt = lambda k, dflt: a[a.index(k) + 1] if k in a else dflt
    micro = int(opt("--micro", 20))
    d = load(emb_dir)
    prep = prepare(d, micro)
    if cmd == "sweep":
        for t in [float(v) for v in opt("--thresholds", "0.3,0.4,0.5,0.6,0.7,0.8").split(",")]:
            lab = labels_at(prep, len(d["ids"]), t)
            p, ip, k = purity(lab, d["speaker_id"])
            print(json.dumps({"threshold": t, "micro": micro, "purity": round(p, 4), "inverse_purity": round(ip, 4),
                              **dict(zip(("leak_any", "leak_mass"), [round(v, 4) for v in leakage(lab, d["speaker_id"])])),
                              "largest_share": round(collections.Counter(lab).most_common(1)[0][1] / len(lab), 4),
                              "clusters": k, "speakers": len(set(d["speaker_id"])), "passes": p >= 0.9 and ip >= 0.8}),
                  flush=True)
    elif cmd == "assign":
        out, t = a[0], float(opt("--threshold", None))
        lab = labels_at(prep, len(d["ids"]), t)
        with open(out, "w") as f:
            f.write("id\tcluster\tgender\tage_group\tduration\n")
            for i, c, g, ag, du in zip(d["ids"], lab, d["gender"], d["age_group"], d["duration"]):
                f.write(f"{i}\t{c}\t{g}\t{ag}\t{float(du):.2f}\n")
        sizes = collections.Counter(lab)
        print(json.dumps({"clips": len(lab), "clusters": len(sizes), "threshold": t, "micro": micro,
                          "largest": sizes.most_common(3), "singletons": sum(1 for v in sizes.values() if v == 1)}))


if __name__ == "__main__":
    main()
