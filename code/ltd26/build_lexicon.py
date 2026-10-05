import collections, sqlite3, sys, unicodedata


def main():
    src, out = sys.argv[1], sys.argv[2]
    counts = {"read": collections.Counter(), "spon": collections.Counter()}
    with open(src, encoding="utf-8") as f:
        for line in f:
            path, _, text = line.rstrip("\n").partition("|")
            part = path.split("/", 2)[1]
            if part in counts:
                counts[part].update(unicodedata.normalize("NFC", text).lower().split())
    words = set(counts["read"]) | set(counts["spon"])
    con = sqlite3.connect(out)
    con.execute("drop table if exists lex")
    con.execute("create table lex (word text primary key, read integer, spon integer)")
    con.executemany("insert into lex values (?, ?, ?)",
                    ((w, counts["read"][w], counts["spon"][w]) for w in words))
    con.commit()
    print({"types": len(words), "read_tokens": sum(counts["read"].values()),
           "spon_tokens": sum(counts["spon"].values())})


if __name__ == "__main__":
    main()
