import glob, json, os, sys, time


def main():
    d, cid, pos, old, new, reason = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    for f in glob.glob(os.path.join(d, "out", "chunk_*.jsonl")):
        lines = open(f, encoding="utf-8").read().splitlines()
        for k, l in enumerate(lines):
            g = json.loads(l)
            if g["id"] != cid:
                continue
            words = g["std"].split()
            if words[pos] != old:
                sys.exit(f"{cid} word {pos} is {words[pos]!r}, not {old!r}")
            words[pos] = new
            g["std"] = " ".join(words)
            lines[k] = json.dumps(g, ensure_ascii=False)
            open(f, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            with open(os.path.join(d, "fixes.jsonl"), "a", encoding="utf-8") as log:
                log.write(json.dumps({"id": cid, "pos": pos, "from": old, "to": new, "by": "review",
                                      "reason": reason, "at": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
            print(f"{cid}: {old} -> {new}")
            return
    sys.exit(f"{cid} not found")


if __name__ == "__main__":
    main()
