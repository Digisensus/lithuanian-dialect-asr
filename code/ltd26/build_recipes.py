import collections, csv, glob, hashlib, json, os

import registry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SPLITS = os.path.join(ROOT, "splits", "v1")
RECIPES = os.path.join(ROOT, "recipes")
DIAL = "../lithuanian-dialect-speech-liepa-3-100h-punctuated/data"
SEED = 2026

PARTS = {
    "S": ("phone.train.v1", "splits/v1/phone.train.txt", "phone.spoken.v1"),
    "X80": ("xspon.train.v1", "splits/v1/xspon.train.txt", "xspon.spoken.v1"),
}
RECIPE_IDS = ["S", "S+D12.dial", "S+D25.dial", "S+D50.dial", "S+D80.dial",
              "S+X80",
              "S+D25.std", "S+D50.std", "S+D80.std",
              "D80.dial", "D80.std",
              "S+D80-A.std", "S+D80-D.std", "S+D80-S.std", "S+D80-Z.std"]
DIAL_LAYER = {"dial": "dial.dialect.v1", "std": "dial.standard.v2"}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def hours_of(ids_file, dur):
    return round(sum(dur[i] for i in open(os.path.join(ROOT, ids_file)).read().split()) / 3600, 2)


def main():
    import pyarrow.parquet as pq
    dur = {}
    for f in glob.glob(os.path.join(DIAL, "*.parquet")):
        t = pq.read_table(f, columns=["id", "duration"]).to_pydict()
        dur.update(zip(t["id"], t["duration"]))
    for f in ("phone.train.txt", "xspon.train.txt"):
        assert os.path.exists(os.path.join(SPLITS, f)), f
    phone_h = {r["split_id"]: float(r["hours"]) for r in registry.rows("splits")}
    xstats = json.load(open(os.path.join(SPLITS, "xspon_stats.json")))

    train = open(os.path.join(SPLITS, "dial.train.txt")).read().split()
    by_spk = collections.defaultdict(list)
    for i in train:
        by_spk[i[:3]].append(i)
    order = [l.split()[0] for l in open(os.path.join(SPLITS, "dose_order.txt")) if l.strip()]
    assert sorted(order) == sorted(by_spk), "dose_order must list exactly the dial.train speakers"
    cum, subsets = 0.0, {}
    for k, spk in enumerate(order):
        cum += sum(dur[i] for i in by_spk[spk]) / 3600
        for n in (12, 25, 50):
            if cum >= n and f"D{n}" not in subsets:
                subsets[f"D{n}"] = order[:k + 1]
    subsets["D80"] = order
    for r in "ADSZ":
        subsets[f"D80-{r}"] = [s for s in order if s[0] != r]
    dose_files = {}
    for name, spks in subsets.items():
        ids = sorted(i for s in spks for i in by_spk[s])
        path = f"splits/v1/dial.{name}.txt"
        with open(os.path.join(ROOT, path), "w") as f:
            f.write("\n".join(ids) + "\n")
        dose_files[name] = (path, len(spks), len(ids), hours_of(path, dur))
    for small, big in (("D12", "D25"), ("D25", "D50"), ("D50", "D80")):
        assert set(subsets[small]) <= set(subsets[big])

    os.makedirs(RECIPES, exist_ok=True)
    rows = []
    for rid in RECIPE_IDS:
        base, _, target = rid.partition(".")
        parts = []
        for code in base.split("+"):
            if code in PARTS:
                split, path, layer = PARTS[code]
                h = phone_h[split] if code == "S" else xstats["hours"]
                parts.append({"code": code, "split": split, "file": path, "layer": layer, "hours": h})
            else:
                path, n_spk, n_clips, h = dose_files[code]
                parts.append({"code": code, "split": "dial.train.v1", "subset": code, "file": path,
                              "layer": DIAL_LAYER[target], "speakers": n_spk, "clips": n_clips, "hours": h})
        recipe = {"recipe_id": rid, "parts": parts, "train_id_files": [p["file"] for p in parts],
                  "sampling_seed": SEED, "builder": "code/ltd26/build_recipes.py"}
        out = os.path.join(RECIPES, rid.replace("/", "_") + ".json")
        with open(out, "w") as f:
            json.dump(recipe, f, indent=1)
        registry.check_recipe(out)
        rows.append({"recipe_id": rid,
                     "train_splits": ";".join(f'{p["split"]}:{p["subset"]}' if "subset" in p else p["split"]
                                              for p in parts),
                     "layer_per_dataset": ";".join(p["layer"] for p in parts),
                     "hours_per_part": ";".join(f'{p["code"]}={p["hours"]}' for p in parts),
                     "sampling_seed": SEED, "recipe_sha256": sha(out)})
    path = os.path.join(ROOT, "registry", "recipes.csv")
    header = next(csv.reader(open(path)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    names = sorted([os.path.basename(v[0]) for v in dose_files.values()] +
                   ["xspon.train.txt", "xspon.train.meta.tsv", "xspon_stats.json"])
    with open(os.path.join(SPLITS, "MANIFEST.t15.sha256"), "w") as f:
        f.writelines(f"{sha(os.path.join(SPLITS, n))}  {n}\n" for n in names)
    print(json.dumps({k: {"speakers": v[1], "clips": v[2], "hours": v[3]} for k, v in dose_files.items()}, indent=1))
    print(json.dumps([(r["recipe_id"], r["hours_per_part"]) for r in rows], indent=0, ensure_ascii=False))


if __name__ == "__main__":
    main()
