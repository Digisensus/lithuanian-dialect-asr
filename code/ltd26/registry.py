import csv, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
REG = os.path.join(ROOT, "registry")
SPLITS = os.path.join(ROOT, "splits", "v1")
HELD_OUT = ("test", "dev", "seen", "excluded")


class LeakageError(Exception):
    pass


def rows(name):
    with open(os.path.join(REG, f"{name}.csv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append(name, row):
    path = os.path.join(REG, f"{name}.csv")
    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    missing = set(row) - set(header)
    if missing:
        raise ValueError(f"unknown columns for {name}: {sorted(missing)}")
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)


def held_out():
    ids = set()
    for part in HELD_OUT:
        ids.update(open(os.path.join(SPLITS, f"dial.{part}.txt")).read().split())
    for part in ("test", "dev", "excluded"):
        path = os.path.join(SPLITS, f"phone.{part}.txt")
        if os.path.exists(path):
            ids.update(open(path).read().split())
    speakers = {r["speaker_id"] for r in csv.DictReader(open(os.path.join(SPLITS, "speakers.csv"), encoding="utf-8"))
                if r["role"] in ("test", "dev")}
    return ids, speakers


def check_train_ids(train_ids):
    ids, speakers = held_out()
    bad = [i for i in train_ids if i in ids]
    bad_spk = sorted({i[:3] for i in train_ids if i[:3] in speakers})
    if bad or bad_spk:
        raise LeakageError(f"{len(bad)} held-out clips (e.g. {bad[:3]}), dev/test speakers {bad_spk[:10]}")
    return True


def check_recipe(path):
    r = json.load(open(path, encoding="utf-8"))
    ids = []
    for p in r.get("train_id_files", []):
        ids += open(os.path.join(ROOT, p)).read().split()
    check_train_ids(ids)
    return len(ids)


def verify_splits():
    bad = []
    lines = [l for m in sorted(glob.glob(os.path.join(SPLITS, "MANIFEST*.sha256"))) for l in open(m)]
    for line in lines:
        digest, name = line.split()
        h = hashlib.sha256(open(os.path.join(SPLITS, name), "rb").read()).hexdigest()
        if h != digest:
            bad.append(name)
    return bad


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check-recipe":
        print(json.dumps({"recipe": sys.argv[2], "train_clips": check_recipe(sys.argv[2]), "leakage": "none"}))
    elif cmd == "verify-splits":
        bad = verify_splits()
        print(json.dumps({"ok": not bad, "changed": bad}))
        sys.exit(1 if bad else 0)
