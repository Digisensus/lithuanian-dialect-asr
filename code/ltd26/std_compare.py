import argparse, glob, json, os


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--chunks")
    ap.add_argument("--show", type=int, default=12)
    x = ap.parse_args()
    names = x.chunks.split(",") if x.chunks else sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(x.work, x.b, "chunk_*.jsonl")))
    clips = same_clip = words = changed = agree = failed = 0
    ex = []
    for n in names:
        src = load(os.path.join(x.work, "in", n + ".jsonl"))
        A, B = load(os.path.join(x.work, x.a, n + ".jsonl")), load(os.path.join(x.work, x.b, n + ".jsonl"))
        for i in B:
            if i not in A:
                continue
            clips += 1
            failed += bool(B[i].get("failed"))
            d, sa, sb = src[i]["dial"].split(), A[i]["std"].split(), B[i]["std"].split()
            same_clip += sa == sb
            if len(sa) != len(d) or len(sb) != len(d):
                continue
            for dw, wa, wb in zip(d, sa, sb):
                words += 1
                if wa != dw or wb != dw:
                    changed += 1
                    agree += wa == wb
                    if wa != wb and len(ex) < x.show:
                        ex.append(f"{dw} → {x.a}: {wa} | {x.b}: {wb}   ({src[i]['dial'][:70]})")
    print(json.dumps({"clips": clips, "failed_b": failed, "clip_exact_pct": round(100 * same_clip / max(1, clips), 1),
                      "changed_words": changed, "agree_on_changed_pct": round(100 * agree / max(1, changed), 1),
                      "word_agree_pct": round(100 * (words - changed + agree) / max(1, words), 2)}))
    for e in ex:
        print("  ", e)


if __name__ == "__main__":
    main()
