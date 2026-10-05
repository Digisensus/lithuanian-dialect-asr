import os, sqlite3, sys, unicodedata

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "std_lexicon.sqlite")


def main():
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    for w in sys.argv[1:]:
        w = unicodedata.normalize("NFC", w).lower()
        row = con.execute("select read, spon from lex where word = ?", (w,)).fetchone()
        print(f"{w}\tread={row[0] if row else 0}\tspon={row[1] if row else 0}")


if __name__ == "__main__":
    main()
