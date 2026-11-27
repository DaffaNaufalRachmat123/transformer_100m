#!/usr/bin/env python3
"""Turn the holdout-clean Wikidata TSVs (expanded_train + tier1) into EN/JA fact-QA SFT rows.

Covers all 41 factual categories injected into corpus_pretrain_v8 (29 expanded + 12 tier1),
explicitly excludes every QID/pair in data/eval/facts_holdout_{entity,property}.jsonl,
and renders natural + contrastive templates matching build_fact_injection_corpus.py.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_EN = "You are a helpful, accurate multilingual assistant."
SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"

# category -> (en_questions, en_answers, ja_questions, ja_answers)
SPECS = {
    # --- Tier 1 core world facts ---
    "capital": (
        ["What is the capital of {sub}?", "Which city is the capital of {sub}?", "Tell me the capital of {sub}."],
        ["The capital of {sub} is {val}.", "{val} is the capital of {sub}."],
        ["{sub}の首都はどこですか？", "{sub}の首都は何ですか？", "{sub}の首都を教えてください。"],
        ["{sub}の首都は{val}です。", "{sub}については、首都は{val}です。"],
    ),
    "capital_admin": (
        ["What is the capital of {sub}?", "Which city serves as the capital of {sub}?", "Tell me the capital of {sub}."],
        ["The capital of {sub} is {val}.", "For {sub}, the capital is {val}."],
        ["{sub}の州都はどこですか？", "{sub}の中心都市はどこですか？", "{sub}の首都はどこですか？"],
        ["{sub}の州都は{val}です。", "{sub}については、州都は{val}です。"],
    ),
    "currency": (
        ["What is the currency of {sub}?", "Which currency does {sub} use?", "Tell me the official currency of {sub}."],
        ["The currency of {sub} is the {val}.", "{sub} uses the {val} as its currency."],
        ["{sub}の通貨は何ですか？", "{sub}ではどの通貨が使われていますか？"],
        ["{sub}の通貨は{val}です。", "{sub}では{val}が使われています。"],
    ),
    "official_language": (
        ["What is an official language of {sub}?", "Name one official language of {sub}.", "Which language is official in {sub}?"],
        ["An official language of {sub} is {val}.", "{sub} has {val} as an official language."],
        ["{sub}の公用語は何ですか？", "{sub}の公用語を一つ挙げてください。"],
        ["{sub}の公用語は{val}です。", "{sub}の公用語の一つは{val}です。"],
    ),
    "population": (
        ["What is the population of {sub}?", "How many people live in {sub}?"],
        ["The population of {sub} is about {val}.", "About {val} people live in {sub}."],
        ["{sub}の人口はどれくらいですか？", "{sub}の人口は何人ですか？"],
        ["{sub}の人口は約{val}人です。", "{sub}にはおよそ{val}人が住んでいます。"],
    ),
    "highest_point": (
        ["What is the highest point of {sub}?", "Which peak is the tallest in {sub}?"],
        ["The highest point of {sub} is {val}.", "{val} is the highest point of {sub}."],
        ["{sub}の最高地点はどこですか？", "{sub}で最も高い山はどこですか？"],
        ["{sub}の最高地点は{val}です。", "{val}は{sub}の最高峰です。"],
    ),
    "continent_of": (
        ["Which continent is {sub} in?", "What continent does {sub} belong to?"],
        ["{sub} is located in {val}.", "{sub} is a country in {val}."],
        ["{sub}はどの大陸にありますか？", "{sub}が属する大陸はどこですか？"],
        ["{sub}は{val}にあります。", "{sub}は{val}に位置する国です。"],
    ),
    "area": (
        ["What is the area of {sub}?", "How large is {sub} in square kilometers?"],
        ["The area of {sub} is about {val} square kilometers.", "{sub} covers roughly {val} square kilometers."],
        ["{sub}の面積はどれくらいですか？", "{sub}の面積を教えてください。"],
        ["{sub}の面積は約{val}平方キロメートルです。"],
    ),
    "element_symbol": (
        ["What is the chemical symbol of {sub}?", "Give the element symbol for {sub}."],
        ["The chemical symbol of {sub} is {val}.", "{sub} has the chemical symbol {val}."],
        ["{sub}の元素記号は何ですか？", "{sub}の化学記号を教えてください。"],
        ["{sub}の元素記号は{val}です。"],
    ),
    "element_number": (
        ["What is the atomic number of {sub}?", "Give the atomic number of {sub}."],
        ["The atomic number of {sub} is {val}.", "{sub} has the atomic number {val}."],
        ["{sub}の原子番号は何ですか？", "{sub}の原子番号を教えてください。"],
        ["{sub}の原子番号は{val}です。"],
    ),
    "driving_side": (
        ["Which side of the road do people drive on in {sub}?", "What is the driving side in {sub}?"],
        ["In {sub}, people drive on the {val}.", "Traffic in {sub} drives on the {val}."],
        ["{sub}はどちら側通行ですか？", "{sub}では車はどちら側を走りますか？"],
        ["{sub}では{val}側通行です。"],
    ),
    "founding_date": (
        ["When was {sub} founded?", "What is the founding date of {sub}?"],
        ["{sub} was founded on {val}.", "The founding date of {sub} is {val}."],
        ["{sub}はいつ設立されましたか？", "{sub}の建国日はいつですか？"],
        ["{sub}は{val}に設立されました。", "{sub}の設立日は{val}です。"],
    ),
    # --- Expanded 29 categories ---
    "person_country": (
        ["Where is {sub} from?", "What is a country of citizenship of {sub}?", "Give one nationality of {sub}."],
        ["{sub} is from {val}.", "One country of citizenship of {sub} is {val}."],
        ["{sub}はどこの国の人ですか？", "{sub}の国籍はどこですか？"],
        ["{sub}は{val}出身です。", "{sub}の国籍は{val}です。"],
    ),
    "person_birthyear": (
        ["When was {sub} born?", "What is the birth year of {sub}?"],
        ["{sub} was born in {val}.", "The birth year of {sub} is {val}."],
        ["{sub}はいつ生まれましたか？", "{sub}の生年は何年ですか？"],
        ["{sub}は{val}年に生まれました。", "{sub}の生年は{val}年です。"],
    ),
    "person_deathyear": (
        ["When did {sub} die?", "What is the death year of {sub}?"],
        ["{sub} died in {val}.", "The death year of {sub} is {val}."],
        ["{sub}はいつ亡くなりましたか？", "{sub}の没年は何年ですか？"],
        ["{sub}は{val}年に亡くなりました。", "{sub}の没年は{val}年です。"],
    ),
    "person_birthplace": (
        ["Where was {sub} born?", "What is the birthplace of {sub}?"],
        ["{sub} was born in {val}.", "The birthplace of {sub} is {val}."],
        ["{sub}の出身地はどこですか？", "{sub}はどこで生まれましたか？"],
        ["{sub}の出身地は{val}です。", "{sub}は{val}で生まれました。"],
    ),
    "person_educated": (
        ["Where did {sub} study?", "Which institution educated {sub}?"],
        ["{sub} studied at {val}.", "{sub} was educated at {val}."],
        ["{sub}はどこで学びましたか？", "{sub}の出身校はどこですか？"],
        ["{sub}は{val}で学びました。", "{sub}は{val}の出身です。"],
    ),
    "person_occupation": (
        ["What does {sub} do?", "Name one occupation of {sub}.", "What is one occupation of {sub}?"],
        ["One occupation of {sub} is {val}.", "{sub} works as a {val}."],
        ["{sub}の職業は何ですか？", "{sub}の職業を一つ挙げてください。"],
        ["{sub}の職業は{val}です。", "{sub}の職業の一つは{val}です。"],
    ),
    "city_country": (
        ["Which country is {sub} in?", "What is the country of {sub}?"],
        ["{sub} is a city in {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の都市ですか？", "{sub}の国はどこですか？"],
        ["{sub}は{val}の都市です。", "{sub}の国は{val}です。"],
    ),
    "city_region": (
        ["Where is {sub} located?", "Which region is {sub} part of?"],
        ["{sub} is located in {val}.", "{sub} is part of {val}."],
        ["{sub}はどこに位置しますか？", "{sub}が属する地域はどこですか？"],
        ["{sub}は{val}に位置します。", "{sub}は{val}の一部です。"],
    ),
    "city_population": (
        ["How many people live in {sub}?", "What is the population of {sub}?"],
        ["The population of {sub} is {val}.", "For {sub}, the population is {val}."],
        ["{sub}の人口は何ですか？", "{sub}の人口はどれくらいですか？"],
        ["{sub}の人口は{val}です。"],
    ),
    "film_director": (
        ["Who directed {sub}?", "Name one director of {sub}."],
        ["{sub} was directed by {val}.", "One director of {sub} is {val}."],
        ["{sub}を監督したのは誰ですか？", "{sub}の監督は誰ですか？"],
        ["{sub}の監督は{val}です。"],
    ),
    "film_year": (
        ["When was {sub} released?", "What is the release year of {sub}?"],
        ["{sub} was released in {val}.", "The release year of {sub} is {val}."],
        ["{sub}はいつ公開されましたか？", "{sub}の公開年は何ですか？"],
        ["{sub}は{val}年に公開されました。", "{sub}の公開年は{val}です。"],
    ),
    "book_author": (
        ["Who wrote {sub}?", "Name one author of {sub}."],
        ["{sub} was written by {val}.", "One author of {sub} is {val}."],
        ["{sub}を書いたのは誰ですか？", "{sub}の著者は誰ですか？"],
        ["{sub}の著者は{val}です。"],
    ),
    "band_origin": (
        ["Where did {sub} form?", "What is the origin of {sub}?"],
        ["{sub} formed in {val}.", "The origin of {sub} is {val}."],
        ["{sub}はどこで結成されましたか？", "{sub}の出身地はどこですか？"],
        ["{sub}は{val}で結成されました。", "{sub}の出身地は{val}です。"],
    ),
    "river_country": (
        ["Which country does the {sub} flow through?", "What is the country of {sub}?"],
        ["The {sub} flows through {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の川ですか？", "{sub}が流れる国はどこですか？"],
        ["{sub}は{val}を流れる川です。", "{sub}の国は{val}です。"],
    ),
    "river_length": (
        ["How long is the {sub}?", "What is the length of {sub}?"],
        ["The length of {sub} is {val}.", "For {sub}, the length is {val}."],
        ["{sub}の長さは何ですか？", "{sub}の全長はどれくらいですか？"],
        ["{sub}の長さは{val}です。"],
    ),
    "mountain_country": (
        ["Which country is {sub} located in?", "What is the country of {sub}?"],
        ["{sub} is located in {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の山ですか？", "{sub}はどこにありますか？"],
        ["{sub}は{val}にある山です。", "{sub}の国は{val}です。"],
    ),
    "mountain_elevation": (
        ["How high is {sub}?", "What is the elevation of {sub}?"],
        ["The elevation of {sub} is {val}.", "For {sub}, the elevation is {val}."],
        ["{sub}の標高は何ですか？", "{sub}の標高はどれくらいですか？"],
        ["{sub}の標高は{val}です。"],
    ),
    "mountain_range": (
        ["Which mountain range does {sub} belong to?", "What range is {sub} part of?"],
        ["{sub} is part of the {val} range.", "{sub} belongs to {val}."],
        ["{sub}はどの山脈に属しますか？", "{sub}の山脈はどこですか？"],
        ["{sub}は{val}の一部です。", "{sub}は{val}に属します。"],
    ),
    "island_country": (
        ["Which country does {sub} belong to?", "What is the country of {sub}?"],
        ["{sub} is an island of {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の島ですか？", "{sub}が属する国はどこですか？"],
        ["{sub}は{val}の島です。", "{sub}の国は{val}です。"],
    ),
    "lake_country": (
        ["Which country is {sub} in?", "What is the country of {sub}?"],
        ["{sub} is a lake in {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の湖ですか？", "{sub}がある国はどこですか？"],
        ["{sub}は{val}にある湖です。", "{sub}の国は{val}です。"],
    ),
    "airport_country": (
        ["Which country is {sub} in?", "What is the country of {sub}?"],
        ["{sub} is an airport in {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の空港ですか？", "{sub}がある国はどこですか？"],
        ["{sub}は{val}にある空港です。", "{sub}の国は{val}です。"],
    ),
    "company_country": (
        ["Where is {sub} based?", "Which country is {sub} from?"],
        ["{sub} is a company based in {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の企業ですか？", "{sub}の国はどこですか？"],
        ["{sub}は{val}の企業です。", "{sub}の国は{val}です。"],
    ),
    "company_hq": (
        ["Where is {sub} headquartered?", "In which city is the headquarters of {sub}?"],
        ["{sub} is headquartered in {val}.", "The headquarters of {sub} is in {val}."],
        ["{sub}の本社はどこにありますか？", "{sub}の本社所在地はどこですか？"],
        ["{sub}の本社は{val}にあります。"],
    ),
    "company_founded": (
        ["When was {sub} founded?", "In what year was {sub} established?"],
        ["{sub} was founded in {val}.", "{sub} was established in {val}."],
        ["{sub}はいつ設立されましたか？", "{sub}の創業年はいつですか？"],
        ["{sub}は{val}年に設立されました。", "{sub}の創業年は{val}です。"],
    ),
    "company_industry": (
        ["What industry does {sub} operate in?", "Which sector is {sub} in?"],
        ["{sub} operates in the {val} industry.", "{sub} is active in {val}."],
        ["{sub}はどの業界の企業ですか？", "{sub}の業種は何ですか？"],
        ["{sub}の業種は{val}です。", "{sub}は{val}業界で事業を展開しています。"],
    ),
    "university_country": (
        ["Which country is {sub} located in?", "What is the country of {sub}?"],
        ["{sub} is a university in {val}.", "The country of {sub} is {val}."],
        ["{sub}はどこの国の大学ですか？", "{sub}がある国はどこですか？"],
        ["{sub}は{val}にある大学です。", "{sub}の国は{val}です。"],
    ),
    "species_parent": (
        ["What parent taxon does {sub} belong to?", "Which group is {sub} classified under?"],
        ["{sub} belongs to {val}.", "The parent taxon of {sub} is {val}."],
        ["{sub}の親分類群は何ですか？", "{sub}はどの分類群に属しますか？"],
        ["{sub}は{val}に属します。", "{sub}の親分類群は{val}です。"],
    ),
    "species_rank": (
        ["What is the taxonomic rank of {sub}?", "Which taxonomic rank does {sub} have?"],
        ["The taxonomic rank of {sub} is {val}.", "{sub} is ranked as {val}."],
        ["{sub}の分類階級は何ですか？"],
        ["{sub}の分類階級は{val}です。"],
    ),
    "star_constellation": (
        ["Which constellation is {sub} in?", "What constellation does {sub} belong to?"],
        ["{sub} is in the constellation {val}.", "{sub} belongs to {val}."],
        ["{sub}はどこの星座にありますか？", "{sub}が属する星座は何ですか？"],
        ["{sub}は{val}座に属します。", "{sub}は星座{val}にあります。"],
    ),
    "planet_star": (
        ["Which star does {sub} orbit?", "What is the host star of {sub}?"],
        ["{sub} orbits {val}.", "The host star of {sub} is {val}."],
        ["{sub}の主星は何ですか？", "{sub}はどの恒星を周回しますか？"],
        ["{sub}は恒星{val}を周回します。", "{sub}の主星は{val}です。"],
    ),
}

DRIVING_EN = {"left": "left", "right": "right", "left-hand traffic": "left", "right-hand traffic": "right"}
DRIVING_JA = {"left": "左", "right": "右", "左手交通": "左", "右手交通": "右", "左側通行": "左", "右側通行": "右"}


def load_holdout_blocklist():
    qids, pairs = set(), set()
    for fn in ("facts_holdout_entity.jsonl", "facts_holdout_property.jsonl"):
        p = ROOT / "data/eval" / fn
        if not p.exists():
            continue
        for l in p.open(encoding="utf-8"):
            d = json.loads(l)
            if d.get("fact_id"):
                qids.add(d["fact_id"])
            if d.get("category") and d.get("qid"):
                qids.add(f"{d['category']}::{d['qid']}")
            se, ve = d.get("subject_en", "").strip(), d.get("value_en", "").strip()
            if se and ve:
                pairs.add((se, ve))
    return qids, pairs


def render_fact(rec, rng, per_fact):
    cat = rec["category"]
    q_en, a_en, q_ja, a_ja = SPECS[cat]
    out = []
    for lang, qs, ans, sysmsg in (
        ("en", q_en, a_en, SYS_EN),
        ("ja", q_ja, a_ja, SYS_JA),
    ):
        sub = rec["subject_en"] if lang == "en" else rec["subject_ja"]
        val = rec["value_en"] if lang == "en" else rec["value_ja"]
        if cat == "driving_side":
            val = DRIVING_EN.get(val.lower(), val) if lang == "en" else DRIVING_JA.get(val.lower(), val.replace("側通行", "").replace("通行", ""))
        if not sub or not val:
            continue
        # Avoid duplicate year suffix `1949年年` in Japanese when value already has `年`
        picked = rng.sample(qs, min(per_fact, len(qs)))
        for qt in picked:
            at = rng.choice(ans)
            q_str = qt.format(sub=sub, val=val)
            a_str = at.format(sub=sub, val=val).replace("年年", "年").replace("座座", "座")
            out.append({
                "text": f"<|system|>{sysmsg}<|user|>{q_str}<|assistant|>{a_str}<|end|>",
                "lang": lang,
                "task": "fact_qa",
                "category": cat,
                "subject_en": rec["subject_en"],
                "subject_ja": rec["subject_ja"],
                "value_en": rec["value_en"],
                "value_ja": rec["value_ja"],
                "fact_id": rec["fact_id"],
                "source": rec["source"],
                "family": "facts",
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded_tsv", default="data/wikidata_facts_expanded_train.tsv")
    ap.add_argument("--tier1_tsv", default="data/wikidata_facts_tier1.tsv")
    ap.add_argument("--out", default="sft_data/sft_facts_expanded")
    ap.add_argument("--expanded_per_fact", type=int, default=2)
    ap.add_argument("--tier1_per_fact", type=int, default=3,
                    help="tier1 core world facts get slightly richer template coverage")
    ap.add_argument("--eval_facts", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    block_qids, block_pairs = load_holdout_blocklist()
    seen_ids = set()
    facts = []
    skipped = Counter()

    for label, rel_path, per_f in (
        ("tier1", a.tier1_tsv, a.tier1_per_fact),
        ("expanded", a.expanded_tsv, a.expanded_per_fact),
    ):
        p = ROOT / rel_path
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                skipped["short_row"] += 1
                continue
            cat, s_en, s_ja, _s_id, v_en, v_ja, _v_id, qid = [c.strip() for c in cols[:8]]
            if cat not in SPECS:
                skipped[f"unknown:{cat}"] += 1
                continue
            if not (s_en and s_ja and v_en and v_ja):
                skipped["missing_field"] += 1
                continue
            if cat == "element_number" and v_en.isdigit() and int(v_en) > 118:
                skipped["superheavy_element"] += 1
                continue
            if cat == "official_language":
                for pref in ("bahasa ", "Bahasa "):
                    if v_en.startswith(pref): v_en = v_en[len(pref):]
                    if v_ja.startswith(pref): v_ja = v_ja[len(pref):]
            fid = f"{cat}::{qid or s_en}"
            if fid in block_qids or (s_en, v_en) in block_pairs:
                skipped["blocked_holdout"] += 1
                continue
            if fid in seen_ids:
                skipped["dup_fact_id"] += 1
                continue
            seen_ids.add(fid)
            facts.append({
                "category": cat, "subject_en": s_en, "subject_ja": s_ja,
                "value_en": v_en, "value_ja": v_ja, "fact_id": fid,
                "source": f"facts_{label}", "_per_fact": per_f,
            })

    rng.shuffle(facts)
    n_eval = min(a.eval_facts, len(facts) // 15)
    eval_facts, train_facts = facts[:n_eval], facts[n_eval:]

    tr = [r for fct in train_facts for r in render_fact(fct, rng, fct["_per_fact"])]
    ev = [r for fct in eval_facts for r in render_fact(fct, rng, 1)]
    rng.shuffle(tr)
    rng.shuffle(ev)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "facts_total": len(facts),
        "facts_train": len(train_facts),
        "facts_eval": len(eval_facts),
        "rows_train": len(tr),
        "rows_eval": len(ev),
        "lang": dict(Counter(r["lang"] for r in tr).most_common()),
        "categories": len(Counter(r["category"] for r in tr)),
        "skipped": dict(skipped.most_common(10)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FACTS-41] unique_facts={len(facts):,} -> train={len(tr):,} eval={len(ev):,} skipped={dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
