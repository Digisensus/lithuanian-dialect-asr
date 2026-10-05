import os, sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ltd26"))
import registry


def test_splits_unchanged():
    assert registry.verify_splits() == []


def test_train_split_is_clean():
    ids = open(os.path.join(registry.SPLITS, "dial.train.txt")).read().split()
    assert registry.check_train_ids(ids)


def test_held_out_clip_is_refused():
    test_id = open(os.path.join(registry.SPLITS, "dial.test.txt")).read().split()[0]
    with pytest.raises(registry.LeakageError):
        registry.check_train_ids([test_id])


def test_dev_speaker_is_refused():
    excluded_id = open(os.path.join(registry.SPLITS, "dial.excluded.txt")).read().split()[0]
    with pytest.raises(registry.LeakageError):
        registry.check_train_ids([excluded_id])


def test_phone_held_out_is_refused():
    for part in ("test", "dev", "excluded"):
        phone_id = open(os.path.join(registry.SPLITS, f"phone.{part}.txt")).read().split()[0]
        with pytest.raises(registry.LeakageError):
            registry.check_train_ids([phone_id])


def test_phone_train_is_clean():
    assert registry.check_train_ids(open(os.path.join(registry.SPLITS, "phone.train.txt")).read().split())


def test_all_recipes_pass_guard():
    import glob, json
    files = sorted(glob.glob(os.path.join(registry.ROOT, "recipes", "*.json")))
    assert len(files) == 15
    for f in files:
        assert registry.check_recipe(f) > 0
        r = json.load(open(f))
        assert all(p["file"] in r["train_id_files"] for p in r["parts"])


def test_dose_subsets_nested():
    ids = lambda n: set(open(os.path.join(registry.SPLITS, f"dial.{n}.txt")).read().split())
    assert ids("D12") < ids("D25") < ids("D50") < ids("D80")
    assert ids("D80") == ids("train")


def test_run_ids_are_unique():
    ids = [r["run_id"] for r in registry.rows("runs")]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
