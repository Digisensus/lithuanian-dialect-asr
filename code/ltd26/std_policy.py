import glob, json, os, sys, time

RULES = [
    ("tadu", {"tadu"}, "tada"), ("anys", {"anys"}, "anie"), ("tokis", {"tokis"}, "toks"),
    ("nieks", {"nieks"}, "niekas"), ("biški", {"biški"}, "biškį"),
    ("maž", {"maž", "matyt"}, "gal"),
    ("pirmukart", None, "pirmąkart"), ("vienukart", None, "vienąkart"), ("vienųkart", None, "vienąkart"),
    ("kitukart", None, "kitąkart"),
    ("toj", {"toje"}, "toj"), ("reiškias", {"reiškiasi"}, "reiškias"), ("mum", {"mums"}, "mum"),
    ("jiem", {"jiems"}, "jiem"), ("dešim", {"dešimt"}, "dešim"),
    ("jum", {"jums"}, "jum"), ("mažu", {"mažu", "matyt"}, "gal"),
    ("da", {"da"}, "dar"), ("biskį", {"biskį"}, "biškį"),
    ("iškart", None, "iškart"),
    ("daba", {"daba"}, "dabar"), ("sava", {"sava"}, "savo"), ("mūsu", {"mūsu"}, "mūsų"),
    ("vė", {"vė"}, "vėl"),
    ("tep", {"tep"}, "taip"), ("tėp", {"tėp"}, "taip"), ("iž", {"iž"}, "iš"), ("pri", {"pri"}, "prie"), ("par", {"par"}, "per"), ("visalaik", {"visąlaik", "visalaik"}, "visą+laiką"), ("vislaik", {"visąlaik", "vislaik"}, "visą+laiką"), ("lygtai", {"lygtai"}, "lyg+tai"), ("lygtais", {"lygtais"}, "lyg+tai"), ("būva", {"būva"}, "buvo"), ("vys", {"vys"}, "vis"),
    ("inai", {"inai"}, "jinai"), ("isai", {"isai"}, "jisai"), ("jin", {"jin"}, "ji"), ("aba", {"aba"}, "arba"),
    ("musiet", {"musėt"}, "musiet"),
    ("yr", {"yr"}, "yra"), ("nėr", {"nėra"}, "nėr"), ("nier", {"nėra"}, "nėr"), ("nebier", {"nebėra"}, "nebėr"),
    ("tenais", {"ten", "tenais"}, "tenai"), ("tinais", {"ten", "tenais", "tinais"}, "tenai"),
    ("tįnais", {"ten", "tenais", "tįnais"}, "tenai"), ("tiktais", {"tiktais"}, "tiktai"), ("čionais", {"čia", "čionais"}, "čionai"),
    ("čenais", {"čia", "čionais", "čenais"}, "čionai"), ("cionais", {"čia", "čionais", "cionais"}, "čionai"),
]
STEMS = [("bulb", "bulv"), ("babūt", "bobut"), ("babyt", "bobut"), ("babut", "bobut"), ("mačiūt", "močiut"),
         ("mučiūt", "močiut"), ("mačiut", "močiut"), ("mučiut", "močiut")]


def v08(dw, sw, read, known):
    if "+" in sw:
        return None
    if dw != sw and not dw.endswith(("j", "y")) and sw != "vietoj":
        if sw[-2:] in ("oj", "ėj", "uj") and read(sw + "e") >= 3:
            return sw + "e", None
        if sw.endswith("y") and read(sw + "je") >= 3 and read(sw) < read(sw + "je"):
            return sw + "je", None
    if dw.endswith(("šims", "šimc")) and (sw.endswith("šimt") or sw.endswith("šims")):
        return sw[:-1], None
    if dw.endswith("am") and sw == dw + "e" and len(dw) > 4:
        return dw, None
    if sw.endswith("davam") and dw.endswith(("davam", "dava", "daam")):
        return sw[:-2] + "om", None
    if dw.endswith("us") and sw.endswith("usi") and not sw.endswith("iausi") and len(sw) > 5:
        return sw[:-1], None
    for a, b in STEMS:
        if sw.startswith(a) and known(b + sw[len(a):]):
            return b + sw[len(a):], None
    if dw in AN_FORMS and sw in JIS_TO_ANAS:
        return JIS_TO_ANAS[sw], None
    return None


VOCATIVE = {"mamyt": "mamyt", "mamit": "mamyt", "vaikel": "vaikel", "tėvel": "tėvel", "tievel": "tėvel",
            "babyt": "bobut", "bobut": "bobut", "sesut": "sesut", "brolel": "brolel", "dukrel": "dukrel"}

AN_FORMS = {"ons", "ana", "an", "on", "anėi", "anys", "any", "ony", "anam", "anuo", "anuos", "anims", "anėms", "anon",
            "anodu", "anoudu", "anų", "anu", "anū", "ano", "anus", "anai", "anuom", "anėm"}
JIS_TO_ANAS = {"jis": "anas", "ji": "ana", "jie": "anie", "jos": "anos", "jam": "anam", "jai": "anai", "jo": "ano",
               "jį": "aną", "ją": "aną", "juos": "anuos", "jiems": "aniems", "joms": "anoms", "jų": "anų",
               "juo": "anuo", "ja": "ana", "jiedu": "anuodu", "juodu": "anuodu", "jodvi": "anodvi"}
REVIEW = {"mažnėjo", "datverdino", "inlūši", "jinlendi"}
REVIEW_MERGES = True


DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "std_lexicon.sqlite")


def main():
    import sqlite3
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    read = lambda w: (con.execute("select read from lex where word = ?", (w,)).fetchone() or (0,))[0]
    d, dry = sys.argv[1], "--dry-run" in sys.argv
    src = {}
    for f in glob.glob(os.path.join(d, "in", "chunk_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            x = json.loads(l)
            src[x["id"]] = x["dial"].split()
    log, n = [], 0
    for f in sorted(glob.glob(os.path.join(d, "out", "chunk_*.jsonl"))):
        lines = open(f, encoding="utf-8").read().splitlines()
        for k, l in enumerate(lines):
            g = json.loads(l)
            words, dial, unsure = g["std"].split(), src[g["id"]], set(g.get("unsure", []))
            for p, (dw, sw) in enumerate(zip(dial, words)):
                joined = sw.replace("+", "")
                spon = lambda w: (con.execute("select spon from lex where word = ?", (w,)).fetchone() or (0,))[0]
                if "+" in sw and sw.split("+")[0] not in ("ar", "ir", "o", "bet", "kad", "nu", "ne", "tai") and (read(joined) >= 50 or (sw.startswith("į+") and read(joined) + spon(joined) >= 20)):
                    log.append({"id": g["id"], "pos": p, "from": sw, "to": sw.replace("+", ""), "by": "policy v0.7"})
                    words[p] = sw = sw.replace("+", "")
                if len(dw) >= 4 and dw.endswith("c") and sw.endswith("ti") and "+" not in sw:
                    log.append({"id": g["id"], "pos": p, "from": sw, "to": sw[:-1], "by": "policy v0.7"})
                    words[p] = sw = sw[:-1]
                r8 = v08(dw, sw, read, lambda w: read(w) + spon(w) > 0)
                if r8:
                    log.append({"id": g["id"], "pos": p, "from": sw, "to": r8[0], "by": "policy v0.8"})
                    words[p] = sw = r8[0]
                    if r8[1]:
                        unsure.add(p)
                letters = {"be", "ce", "čė", "de", "dė", "ef", "ge", "gė", "ha", "haš", "je", "jot", "ka", "el", "em", "en", "pe", "er", "es", "eš", "te", "tė",
                           "ū", "ų", "ė", "ę", "į", "y", "vė", "ve", "zė", "žė", "iks"}
                spelled = (p and dial[p - 1] in letters and dial[p - 1] != "į") or (p + 1 < len(dial) and dial[p + 1] in letters)
                for rd, cur, new in RULES:
                    if rd == "vė" and (spelled or any(len(w) > 3 and w.startswith("vė") for w in dial[p + 1:p + 3])):
                        continue
                    if dw == rd and sw == dw and p in unsure:
                        continue
                    if dw == rd and (cur is None or sw in cur) and sw != new:
                        log.append({"id": g["id"], "pos": p, "from": sw, "to": new, "by": "policy v0.4"})
                        words[p] = new
                if dw in ("i", "ka", "tį", "ti", "an", "un", "in", "nieka", "koki", "sak", "given", "buv", "dirb", "turėj", "galvo", "gyven") and sw == dw and p not in unsure:
                    unsure.add(p)
                    log.append({"id": g["id"], "pos": p, "from": sw, "to": sw, "by": "policy v0.7", "note": "ambiguous form left unchanged, marked unsure"})
                if dw in ("mumis", "momis") and sw == "mumis" and not (p and dial[p - 1] in ("su", "sa")) and p not in unsure:
                    unsure.add(p)
                    log.append({"id": g["id"], "pos": p, "from": sw, "to": sw, "by": "policy v0.8.1", "note": "mumis without su: check accusative (mus)"})
                if dw in REVIEW and p not in unsure:
                    unsure.add(p)
                    log.append({"id": g["id"], "pos": p, "from": sw, "to": sw, "by": "policy v0.4", "note": "doubtful merge, marked unsure"})
            g["std"], g["unsure"] = " ".join(words), sorted(unsure)
            lines[k] = json.dumps(g, ensure_ascii=False)
        if not dry:
            open(f, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if not dry:
        with open(os.path.join(d, "fixes.jsonl"), "a", encoding="utf-8") as out:
            for e in log:
                out.write(json.dumps(dict(e, at=time.strftime("%Y-%m-%d %H:%M")), ensure_ascii=False) + "\n")
    changed = [e for e in log if e["from"] != e["to"]]
    print(json.dumps({"changed_words": len(changed), "marked_unsure": len(log) - len(changed),
                      "by_rule": {f"{e['from']}→{e['to']}": sum(1 for x in changed if (x['from'], x['to']) == (e['from'], e['to'])) for e in changed}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
