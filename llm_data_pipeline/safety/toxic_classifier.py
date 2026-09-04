"""
Toxic Content Classifier — per DATA_PIPELINE_PLAN A.7.

Approach:
  1. Per-language keyword + pattern-based detection (fast, no GPU needed)
  2. Optional: fastText toxic classifier per language
  3. Threshold: reject if toxic_score >= 0.7

Categories: severe_toxicity, threat, sexually_explicit, identity_attack,
            insult, profanity, harassment
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

TOXIC_CATEGORIES = [
    "severe_toxicity", "threat", "sexually_explicit",
    "identity_attack", "insult", "profanity", "harassment",
]

TOXIC_PATTERNS_EN = {
    "severe_toxicity": [],
    "threat": [
        r"\b(?:i|we)\s+(?:will|gonna|going\s+to)\s+(?:kill|hurt|harm|destroy|attack|murder)\b",
        r"\b(?:kill|murder|hurt|harm)\s+(?:you|them|him|her|yourself)\b",
        r"\b(?:death|die)\s+(?:threats?|wish)\b",
        r"\b(?:bomb|shoot|terror)\s+(?:threat|attack|plot)\b",
        r"\b(?:i|we)\s+(?:know\s+where\s+you\s+live)\b",
    ],
    "sexually_explicit": [
        r"\b(?:porn(?:ography|hub)?|xxx|sex\s*(?:tape|video))\b",
        r"\b(?:explicit\s*(?:sexual|content|material))\b",
        r"\b(?:nude|naked|topless)\s*(?:photo|pic|image)\b",
    ],
    "identity_attack": [
        r"\b(?:you|they|those)\s+(?:dumb|stupid|idiot|moron|retard)\s*\b",
        r"\b(?:(?:racial|ethnic)\s+slur)\b",
        r"\b(?:go\s+back\s+to\s+your\s+country)\b",
        r"\b(?:all\s+(?:men|women|gays?|muslims?|jews?|blacks?|whites?|asians?)\s+are)\b",
    ],
    "insult": [
        r"\b(?:idiot|stupid|dumb|moron|loser|pathetic|worthless)\b",
        r"\b(?:shut\s+(?:the\s+)?(?:fuck|hell)\s+up)\b",
    ],
    "profanity": [],
    "harassment": [
        r"\b(?:kill\s+yourself|kys)\b",
        r"\b(?:self[- ]?harm)\b",
        r"\b(?:go\s+(?:die|kill\s+yourself))\b",
        r"\b(?:nobody\s+(?:loves?|likes?|wants?|cares?\s+about)\s+you)\b",
    ],
}

TOXIC_PATTERNS_ID = {
    "identity_attack": [
        r"\b(?:kafir|munafik|bangsat|bajingan|anjing|babi|monyet)\b",
        r"\b(?:cina|tionghoa|pribumi|pendatang)\s+(?:sialan|bangsat)\b",
    ],
    "insult": [
        r"\b(?:goblok|tolol|bodoh|bego|dungu|geblek)\b",
        r"\b(?:sinting|gila|sarap|edan)\b",
        r"\b(?:brengsek|sialan|kampret|jancok|asu|jangkrik)\b",
        r"\b(?:dasar\s+(?:bangsat|bajingan|brengsek))\b",
    ],
    "threat": [
        r"\b(?:awas\s+(?:ya|lu|kamu|kau|lo))\b",
        r"\b(?:gue|aku|saya)\s+(?:bunuh|hajar|gebuk|bacok)\b",
        r"\b(?:mati\s+(?:lu|kamu|kau|lo|aja|sana))\b",
    ],
}

TOXIC_PATTERNS_JA = {
    "identity_attack": [
        r"\b(?:死ね|消えろ|くたばれ|ぶっ殺す)\b",
        r"\b(?:キチガイ|基地外)\b",
        r"\b(?:在日|チョン|シナ)\b",
    ],
    "insult": [
        r"\b(?:バカ|アホ|マヌケ|ボケ|カス)\b",
        r"\b(?:クズ|ゴミ|カス|雑魚)\b",
        r"\b(?:キモい|ウザい|うざい|ダサい)\b",
    ],
    "threat": [
        r"\b(?:殺す|ぶっ殺す|抹殺)\b",
        r"\b(?:死ね|氏ね|タヒね)\b",
    ],
}

TOXIC_PATTERNS_KO = {
    "identity_attack": [],
    "insult": [
        r"\b(?:바보|멍청이|병신|등신|또라이)\b",
        r"\b(?:미친|미쳤어|돌았어|제정신이야)\b",
        r"\b(?:쓰레기|찌질이|한심해)\b",
    ],
    "threat": [
        r"\b(?:죽일|죽여|죽어|죽을래)\b",
        r"\b(?:조심해|가만안둬|혼내줄게)\b",
    ],
}

LANG_PATTERNS = {
    "en": TOXIC_PATTERNS_EN,
    "id": TOXIC_PATTERNS_ID,
    "ja": TOXIC_PATTERNS_JA,
    "ko": TOXIC_PATTERNS_KO,
}

LANG_PATTERNS["en"]["profanity"] = [
    r"\bfuck(?:ing|ed|er|s)?\b",
    r"\bshit(?:ty|hole|head)?\b",
    r"\bass(?:hole|hat|wipe)?\b",
    r"\bbitch(?:es|ing)?\b",
    r"\bdick(?:head)?\b",
    r"\bcunt\b",
    r"\bmother\s*fucker\b",
]
LANG_PATTERNS["en"]["severe_toxicity"] = [
    r"\bkill\s*(?:yourself|urself)\b",
    r"\b(?:i\s+hope\s+you\s+(?:die|rot|suffer))\b",
]
LANG_PATTERNS["id"]["profanity"] = [
    r"\b(?:kontol|memek|peler|itil|tempek|jembut)\b",
    r"\b(?:ngentot|entot|ngewe|ewe|colmek|coli)\b",
    r"\b(?:jamet|asu|kirik|celeng|wedus|jangkrik)\b",
]
LANG_PATTERNS["id"]["severe_toxicity"] = [
    r"\b(?:bunuh\s+diri|mati\s+saja|mampus\s+lu)\b",
]
LANG_PATTERNS["id"]["sexually_explicit"] = [
    r"\b(?:bokep|porno|sex|ngentot|pemuasan)\b",
]
LANG_PATTERNS["ja"]["profanity"] = [
    r"\b(?:クソ|糞|ふざけんな|ふざけるな)\b",
]
LANG_PATTERNS["ja"]["severe_toxicity"] = [
    r"\b(?:自殺しろ|死んで|消えろ|くたばれ)\b",
]
LANG_PATTERNS["ko"]["profanity"] = [
    r"\b(?:씨발|씨팔|개새끼|좆|젠장|염병)\b",
    r"\b(?:존나|좆나|미친|닥쳐|꺼져)\b",
]
LANG_PATTERNS["ko"]["severe_toxicity"] = [
    r"\b(?:자살해|죽어라|꺼져라)\b",
]
LANG_PATTERNS["ko"]["sexually_explicit"] = [
    r"\b(?:야동|포르노|섹스|성인물)\b",
]


class ToxicClassifier:
    """Detect toxic content across target languages."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.stats = defaultdict(int)

    def score(self, text: str, language: str = "en") -> Tuple[float, Dict[str, float], List[str]]:
        text_lower = text.lower()
        patterns = LANG_PATTERNS.get(language, LANG_PATTERNS["en"])

        category_scores = {}
        all_matches = []

        for category in TOXIC_CATEGORIES:
            cat_patterns = patterns.get(category, [])
            matches = 0
            for pat in cat_patterns:
                found = re.findall(pat, text_lower)
                matches += len(found)
                all_matches.extend(found)

            max_possible = max(1, len(text_lower.split()) / 10)
            category_scores[category] = min(1.0, matches / max_possible)

        overall = max(category_scores.values()) if category_scores else 0.0
        return overall, category_scores, all_matches

    def is_toxic(self, text: str, language: str = "en") -> Tuple[bool, float]:
        overall, _, _ = self.score(text, language)
        return overall >= self.threshold, overall

    def filter_documents(self, documents: List[Dict], text_field: str = "text",
                         language_field: str = "language") -> Tuple[List[Dict], List[Dict]]:
        clean = []
        toxic = []
        for doc in documents:
            text = doc.get(text_field, "")
            lang = doc.get(language_field, "en")
            overall, cat_scores, matches = self.score(text, lang)
            doc_out = dict(doc)
            doc_out["toxic_score"] = overall
            doc_out["toxic_categories"] = {k: v for k, v in cat_scores.items() if v > 0}
            doc_out["toxic_matches"] = matches[:20]
            self.stats["total"] += 1
            if overall >= self.threshold:
                self.stats["toxic"] += 1
                toxic.append(doc_out)
            else:
                self.stats["clean"] += 1
                clean.append(doc_out)
        return clean, toxic

    def get_stats(self) -> Dict:
        return dict(self.stats)

    def print_summary(self):
        s = dict(self.stats)
        total = s.get("total", 1)
        print(f"\n{'='*60}\n TOXICITY FILTERING SUMMARY\n{'='*60}")
        print(f"  Documents scanned  : {s.get('total', 0):,}")
        print(f"  Clean              : {s.get('clean', 0):,} ({100*s.get('clean',0)/total:.1f}%)")
        print(f"  Toxic (≥{self.threshold}) : {s.get('toxic', 0):,} ({100*s.get('toxic',0)/total:.1f}%)")
        print("=" * 60)
