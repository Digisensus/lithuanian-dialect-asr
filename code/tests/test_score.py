import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ltd26"))
import score


def test_normalise():
    assert score.normalise("Labas, RYTAS!  Kaip  sekasi?") == "labas rytas kaip sekasi"
    assert score.normalise("é") == score.normalise("é")
    assert score.normalise("2024 m.") == "2024 m"


def test_edit_distance_counts():
    assert score.edit_distance(list("abc"), list("abc")) == (0, 0, 0, 0)
    cost, sub, dele, ins = score.edit_distance("a b c".split(), "a x c d".split())
    assert (cost, sub, dele, ins) == (2, 1, 0, 1)
    assert score.edit_distance("a b".split(), [])[:3] == (2, 0, 2)


def test_either_matches_dialect_or_standard():
    slots = [(("buva",), ("buvo",)), (("gera",), ("gera",))]
    assert score.either_align(slots, "buvo gera".split())[0] == 0
    assert score.either_align(slots, "buva gera".split())[0] == 0
    assert score.either_align(slots, "bova gera".split())[0] == 1


def test_either_multiword_standard():
    slots = [(("ipanevėžio",), ("į", "panevėžio")), (("važiavo",), ("važiavo",))]
    total, per = score.either_align(slots, "į panevėžio važiavo".split())
    assert total == 0 and per == [0, 0]
    assert score.either_align(slots, "ipanevėžio važiavo".split())[0] == 0


def test_either_insertion_and_deletion():
    slots = [(("a",), ("a",)), (("b",), ("b",))]
    assert score.either_align(slots, "a x b".split())[0] == 1
    assert score.either_align(slots, "a".split())[0] == 1
    assert score.either_align(slots, "a b z".split())[0] == 1
    assert score.either_align(slots, [])[0] == 2


def test_clip_scores_views():
    row = {"dial": "tai buva gera", "std": "tai buvo gera",
           "alignment": [["tai", "tai", "same", "O"], ["buva", "buvo", "pron", "O"], ["gera", "gera", "same", "O"]]}
    c = score.clip_scores(row, "Tai buvo gera.")
    assert (c["e_dial"], c["e_std"], c["e_either"]) == (1, 0, 0)
    c = score.clip_scores(row, "tai buva gera")
    assert (c["e_dial"], c["e_std"], c["e_either"]) == (0, 1, 0)
    assert c["tag_n"] == {"same": 2, "pron": 1}


def test_aggregate_and_bootstrap():
    per = {"A01-1": {"n": 10, "n_std": 10, "e_dial": 1, "e_std": 2, "e_either": 1, "sub": 1, "del": 0, "ins": 0,
                     "chars": 50, "e_char": 1, "tag_err": {}, "tag_n": {}},
           "D02-1": {"n": 10, "n_std": 10, "e_dial": 3, "e_std": 3, "e_either": 2, "sub": 3, "del": 0, "ins": 0,
                     "chars": 50, "e_char": 4, "tag_err": {}, "tag_n": {}}}
    r = score.aggregate(per, list(per))
    assert r["wer_dialect"] == 20.0 and r["wer_either"] == 15.0 and r["wer_by_region"] == {"A": 10.0, "D": 30.0}
    lo, hi = score.bootstrap(per, list(per), n_boot=500)
    assert 5.0 <= lo <= hi <= 20.0


def test_single_reference_layer_scores_the_same_in_all_views(tmp_path):
    p = tmp_path / "phone.jsonl"
    p.write_text('{"id": "abc", "text": "labas rytas"}\n', encoding="utf-8")
    row = score.load_layer(str(p))["abc"]
    c = score.clip_scores(row, "Labas, vakaras.")
    assert c["e_dial"] == c["e_std"] == c["e_either"] == 1


def test_either_handles_long_insertions_between_matching_words():
    slots = [(("a",), ("a",)), (("b",), ("b",))]
    for length in range(1, 9):
        hyp = ["a"] + ["x"] * length + ["b"]
        total, per = score.either_align(slots, hyp)
        assert total == score.edit_distance(["a", "b"], hyp)[0] == length
        assert sum(per) == total


def test_leading_insertions_are_included_in_word_class_errors():
    row = {"dial": "a b", "std": "a b",
           "alignment": [["a", "a", "kept", ""], ["b", "b", "same", ""]]}
    c = score.clip_scores(row, "x y z a b")
    assert c["e_either"] == 3
    assert c["tag_err"]["kept"] == 3
    assert sum(c["tag_err"].values()) == c["e_either"]


def test_either_matches_exhaustive_reference_alternatives():
    import itertools
    import random
    rnd = random.Random(91)
    for _ in range(150):
        slots = [(tuple(rnd.choices("abcd", k=rnd.randrange(1, 3))),
                  tuple(rnd.choices("abcd", k=rnd.randrange(1, 3))))
                 for _ in range(rnd.randrange(1, 5))]
        hyp = rnd.choices("abcd", k=rnd.randrange(12))
        expected = min(score.edit_distance([w for alt in combination for w in alt], hyp)[0]
                       for combination in itertools.product(*slots))
        total, per = score.either_align(slots, hyp)
        assert total == expected
        assert sum(per) == total


def test_unknown_speakers_resample_recordings_not_shared_id_prefixes():
    per = {"abc000": {"n": 10, "e_either": 1},
           "abc111": {"n": 10, "e_either": 9}}
    assert score.bootstrap(per, list(per), n_boot=500) == [10.0, 90.0]
    for c in per.values():
        c["spk"] = "same-speaker"
    assert score.bootstrap(per, list(per), n_boot=500) == [50.0, 50.0]


def test_score_file_uses_explicit_speakers_or_split_appropriate_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "load_split", lambda _: ["A01-1", "A01-2"])
    monkeypatch.setattr(score, "load_layer", lambda _: {
        "A01-1": {"dial": "a", "std": "a"},
        "A01-2": {"dial": "b", "std": "b"}})
    p = tmp_path / "hyps.jsonl"
    p.write_text('{"id":"A01-1","hyp":"a"}\n{"id":"A01-2","hyp":"b"}\n')
    _, dialect, _ = score.score_file(str(p), "dial.test.v1", "unused")
    _, telephone, _ = score.score_file(str(p), "phone.test.v1", "unused")
    assert [c["spk"] for c in dialect.values()] == ["A01", "A01"]
    assert [c["spk"] for c in telephone.values()] == ["A01-1", "A01-2"]
