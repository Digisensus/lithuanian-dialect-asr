import json, sys

import numpy as np

import cluster_speakers as C


def nearest_test_sim(x, test_mask):
    import faiss
    idx = faiss.IndexFlatIP(x.shape[1])
    idx.add(x[test_mask])
    sims, _ = idx.search(x[~test_mask], 1)
    return sims[:, 0]


def isolation(x, lab, k=20):
    import faiss
    idx = faiss.IndexFlatIP(x.shape[1])
    idx.add(x)
    sims, nn = idx.search(x, k + 1)
    other = lab[nn] != lab[:, None]
    best = np.where(other, sims, -1).max(axis=1)
    out = {}
    for c in np.unique(lab):
        out[c] = float(best[lab == c].mean())
    return out


def draw_test(lab, share, rng, order=None, cap=None):
    cl, inv = np.unique(lab, return_inverse=True)
    size = np.bincount(inv)
    seq = rng.permutation(len(cl)) if order is None else order
    chosen, n = np.zeros(len(cl), bool), 0
    for c in seq:
        if n >= share * len(lab):
            break
        if cap is not None and size[c] > cap:
            continue
        chosen[c], n = True, n + size[c]
    return chosen[inv]


def main():
    a = sys.argv[2:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    micro, t, share, reps = opt("--micro", 5), opt("--threshold", 0.3), opt("--share", 0.10), opt("--reps", 5)
    select = opt("--select", "random")
    cap_share = opt("--cap", 0.0)
    d = C.load(sys.argv[1])
    x = d["emb"].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    spk = np.asarray(d["speaker_id"])
    prep = C.prepare(d, micro)
    lab = np.asarray(C.labels_at(prep, len(x), t))
    p, ip, k = C.purity(lab, spk)
    order = None
    if select == "isolated":
        iso = isolation(x, lab)
        cl = np.unique(lab)
        order = np.argsort([iso[c] for c in cl])
    taus = [float(v) for v in opt("--taus", "0.5,0.6,0.7,1.01").split(",")]
    res = {tau: [] for tau in taus}
    sp, sinv = np.unique(spk, return_inverse=True)
    tot = np.bincount(sinv)
    for r in range(reps if select == "random" else 1):
        test = draw_test(lab, share, np.random.default_rng(2026 + r), order,
                         cap=cap_share * len(lab) if cap_share else None)
        sim = nearest_test_sim(x, test)
        rest = np.where(~test)[0]
        for tau in taus:
            keep = np.zeros(len(x), bool)
            keep[rest[sim < tau]] = True
            left = np.bincount(sinv[keep], minlength=len(sp))
            ts = sinv[test]
            res[tau].append((float((left[ts] / tot[ts]).mean()), float((left[ts] > 0).mean()),
                             1 - keep.sum() / (~test).sum()))
    for tau in taus:
        m = np.mean(res[tau], axis=0)
        print(json.dumps({"micro": micro, "threshold": t, "select": select, "share": share, "cap": cap_share, "purity": round(p, 3),
                          "inverse_purity": round(ip, 3), "clusters": k,
                          "tau": "none" if tau > 1 else tau, "leak_mass": round(m[0], 4),
                          "leak_any": round(m[1], 4), "train_dropped": round(m[2], 4)}), flush=True)


if __name__ == "__main__":
    main()
