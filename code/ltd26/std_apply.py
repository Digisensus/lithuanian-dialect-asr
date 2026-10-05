import argparse, concurrent.futures as cf, glob, json, os, re, sqlite3, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "data", "std_lexicon.sqlite")
NASAL = {"a": "ą", "e": "ę", "i": "į", "u": "ų"}
SYSTEM = """You help standardise Lithuanian dialect transcripts. A sentence has already been converted
to standard spelling except for some words marked [n]. For each marked word, choose the correct
option by its number. The options are standard Lithuanian forms of the same word; choose the one
the sentence requires grammatically and by meaning:
- nasal endings (ą ę į ų) mark accusative singular (matau dieną, tą vieną ilgą pastatą), genitive
  plural (daug metų, mūsų), and similar forms; without them the word is nominative, etc.;
- `į` is the preposition "to/into" (followed by accusative), `ir` means "and/also";
- `kad` = "that/so that", `ką` = "what" (accusative), `ko` = "what" (genitive), `kai` = "when".
Keep the plain form unless the sentence clearly requires the other one. Nominative subjects,
vocatives (`vaikeli`), names in exclamations (`jėzus marija`) and colloquial short forms (`mum`)
stay as they are.

Examples:
Sentence: [1] vedas mani [2] tvartą          [1] 1) ir 2) į   [2] 1) ir 2) į        -> {"1": 1, "2": 2}
Sentence: tai tą rodos [1] [2] pastatą       [1] 1) viena 2) vieną  [2] 1) ilga 2) ilgą -> {"1": 2, "2": 2}
Sentence: [1] valandų praėjo                 [1] 1) pora 2) porą                        -> {"1": 1}
Sentence: namie darbai [1] ką                 [1] 1) visa 2) visą                        -> {"1": 1}
Sentence: prieš [1]                           [1] 1) galu 2) galą                        -> {"1": 2}
Use the neighbouring sentences for context. Reply with JSON only, for example {"1": 2, "2": 1}."""


def post(url, body, timeout=300, tries=4):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (OSError, ValueError):
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("typemap")
    ap.add_argument("--out", default="out_B")
    ap.add_argument("--chunks")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--think-budget", type=int, default=300)
    ap.add_argument("--before", type=int, default=3)
    ap.add_argument("--after", type=int, default=2)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--no-llm", action="store_true", help="take the first option everywhere (baseline)")
    ap.add_argument("--export-items", help="write the multiple-choice items to this dir (in/part_*.jsonl) and stop")
    ap.add_argument("--parts", type=int, default=2, help="number of item parts with --export-items")
    ap.add_argument("--choices", help="dir with out/part_*.jsonl {id, c} answers (e.g. from agents) instead of the LLM")
    a = ap.parse_args()

    decisions = {d["type"]: d for d in map(json.loads, open(a.typemap, encoding="utf-8"))}
    con = sqlite3.connect(f"file:{os.path.abspath(DB)}?mode=ro", uri=True)
    read = {}

    def std_word(w):
        if w not in read:
            row = con.execute("select read from lex where word = ?", (w,)).fetchone()
            read[w] = row[0] if row else 0
        return read[w] >= 3

    def plan(words):
        out = []
        for w in words:
            d = decisions.get(w)
            if d and d["decision"] in ("std", "merge"):
                out.append((d["std"], None, "T"))
            elif d and d["decision"] == "keep":
                out.append((w, None, "="))
            elif d and d["decision"] == "ctx":
                out.append((None, list(dict.fromkeys(d["cands"])), "C"))
            elif not d and w[-1:] in NASAL and std_word(w) and std_word(w[:-1] + NASAL[w[-1]]) \
                    and read[w[:-1] + NASAL[w[-1]]] >= 0.05 * read[w]:
                out.append((None, [w, w[:-1] + NASAL[w[-1]]], "C"))
            else:
                out.append((w, None, "="))
        return out

    names = a.chunks.split(",") if a.chunks else sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(a.work, "in", "chunk_*.jsonl")))
    jobs = []
    for name in names:
        src = [json.loads(l) for l in open(os.path.join(a.work, "in", name + ".jsonl"), encoding="utf-8")]
        if a.limit:
            src = src[:a.limit]
        for i, r in enumerate(src):
            before = [x["dial"] for x in src[max(0, i - a.before):i] if x["spk"] == r["spk"]]
            after = [x["dial"] for x in src[i + 1:i + 1 + a.after] if x["spk"] == r["spk"]]
            jobs.append((name, i, r, before, after, plan(r["dial"].split())))
    stats = {"clips": 0, "llm_clips": 0, "llm_words": 0, "fallback_words": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def item(r, p, before):
        shown, opts, n = [], [], 0
        for fixed, o, _ in p:
            if o:
                n += 1
                shown.append(f"[{n}]")
                opts.append(o)
            else:
                shown.append(fixed)
        return {"id": r["id"], "d": r["dial"], "s": " ".join(shown), "o": opts, "b": " | ".join(before[-2:])}

    if a.export_items:
        items = [item(r, p, before) for _, _, r, before, _, p in jobs if any(o for _, o, _ in p)]
        os.makedirs(os.path.join(a.export_items, "in"), exist_ok=True)
        os.makedirs(os.path.join(a.export_items, "out"), exist_ok=True)
        size = -(-len(items) // a.parts)
        for k in range(a.parts):
            with open(os.path.join(a.export_items, "in", f"part_{k:02d}.jsonl"), "w", encoding="utf-8") as f:
                for it in items[k * size:(k + 1) * size]:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(json.dumps({"clips": len(jobs), "clips_with_items": len(items), "marked_words": sum(len(i["o"]) for i in items),
                          "parts": a.parts}))
        return
    given = {}
    if a.choices:
        for f in glob.glob(os.path.join(a.choices, "out", "part_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    g = json.loads(l)
                    given[g["id"]] = {j: c for j, c in enumerate(g.get("c", []), 1) if isinstance(c, int)}

    def choose(r, p, before, after):
        marks = [k for k, (_, opts, _) in enumerate(p) if opts]
        shown, n = [], 0
        for k, (fixed, opts, _) in enumerate(p):
            if opts:
                n += 1
                shown.append(f"[{n}]")
            else:
                shown.append(fixed)
        user = ""
        if before:
            user += "Before:\n" + "\n".join(before) + "\n"
        if after:
            user += "After:\n" + "\n".join(after) + "\n"
        user += "\nOriginal dialect sentence: " + r["dial"] + "\nSentence: " + " ".join(shown) + "\n"
        for j, k in enumerate(marks, 1):
            user += f"[{j}] options: " + "  ".join(f"{o}) {w}" for o, w in enumerate(p[k][1], 1)) + "\n"
        body = {"model": a.model, "temperature": 0, "max_tokens": a.think_budget + 40 + 8 * len(marks),
                "thinking_token_budget": a.think_budget,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "chat_template_kwargs": {"enable_thinking": a.think_budget > 0}}
        resp = post(a.url, body)
        stats["prompt_tokens"] += resp["usage"]["prompt_tokens"]
        stats["completion_tokens"] += resp["usage"]["completion_tokens"]
        text = (resp["choices"][0]["message"]["content"] or "").split("</think>")[-1]
        m = re.search(r"\{[^{}]*\}", text)
        try:
            ans = {int(k): int(v) for k, v in json.loads(m.group(0)).items()} if m else {}
        except (ValueError, TypeError):
            ans = {}
        return marks, ans

    def run(job):
        name, i, r, before, after, p = job
        words = r["dial"].split()
        std = [fixed for fixed, _, _ in p]
        srcs = [s for _, _, s in p]
        unsure = []
        if any(opts for _, opts, _ in p):
            if a.no_llm or a.choices:
                marks, ans = [k for k, (_, o, _) in enumerate(p) if o], given.get(r["id"], {})
            else:
                marks, ans = choose(r, p, before, after)
            stats["llm_clips"] += 1
            for j, k in enumerate(marks, 1):
                opts = p[k][1]
                stats["llm_words"] += 1
                c = ans.get(j)
                if c is None or not 1 <= c <= len(opts):
                    std[k], srcs[k] = opts[0], "F"
                    unsure.append(k)
                    stats["fallback_words"] += 1
                else:
                    std[k] = opts[c - 1]
        for k in range(len(words)):
            if srcs[k] == "T" and std[k] == words[k]:
                srcs[k] = "="
        stats["clips"] += 1
        return name, i, {"id": r["id"], "std": " ".join(std), "unsure": unsure, "src": "".join(srcs)}

    t0 = time.time()
    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for name, i, g in ex.map(run, jobs):
            res.setdefault(name, {})[i] = g
    os.makedirs(os.path.join(a.work, a.out), exist_ok=True)
    for name, parts in res.items():
        with open(os.path.join(a.work, a.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for i in sorted(parts):
                f.write(json.dumps(parts[i], ensure_ascii=False) + "\n")
    stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
