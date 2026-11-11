"""Rule engine for L2-SyntaxLens: deterministic diagnostics over Universal Dependencies parses."""
import functools
import html
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import spacy
from pydantic import BaseModel, Field

from .l1 import L1_NOTES


@functools.lru_cache(maxsize=1)
def load_nlp():
    return spacy.load("en_core_web_sm")

# ---------- Schemas ----------
class Issue(BaseModel):
    rule_id: str
    category: str
    sentence_index: int
    token_indices: List[int]
    focus: str
    fix: str
    explanation: str
    l1_note: Optional[str] = None


class Report(BaseModel):
    text: str
    sentence_count: int
    issue_count: int
    issues: List[Issue] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


# ---------- Lexical resources ----------
ALLOWED_PREPS = {
    "depend": {"on", "upon"}, "rely": {"on", "upon"}, "insist": {"on", "upon"},
    "listen": {"to"}, "interested": {"in"}, "wait": {"for", "until", "on"},
    "graduate": {"from", "in"}, "agree": {"with", "to", "on", "about"},
    "consist": {"of", "in"}, "good": {"at", "for", "with", "to"},
}
PREFERRED_PREP = {"depend": "on", "rely": "on", "insist": "on", "listen": "to", "interested": "in",
                  "wait": "for", "graduate": "from", "agree": "with", "consist": "of", "good": "at"}
EXTRA_PREP = {"discuss": "about", "mention": "about", "marry": "with", "emphasize": "on", "approach": "to"}
PREPS = {"on", "of", "in", "at", "to", "for", "from", "with", "by", "about", "upon", "into"}

MASS = {"information", "advice", "research", "furniture", "evidence", "homework",
        "knowledge", "equipment", "luggage", "progress", "feedback", "software"}
PLURAL_QUANT = {"many", "few", "several", "these", "those", "both"}
MASS_QUANT = {"much"}

PAST_WORDS = {"yesterday", "ago", "formerly", "previously"}
LAST_UNITS = {"week", "month", "year", "night", "summer", "winter", "semester", "monday", "tuesday",
              "wednesday", "thursday", "friday", "saturday", "sunday"}

ADJ_RANK = {  # lower rank comes first in English
    **{w: 1 for w in ("big", "large", "small", "huge", "tiny", "tall", "short", "long")},
    **{w: 2 for w in ("old", "new", "young", "ancient", "modern")},
    **{w: 3 for w in ("red", "blue", "green", "black", "white", "yellow", "dark", "brown", "grey", "gray")},
    **{w: 4 for w in ("wooden", "metal", "plastic", "cotton", "silver", "golden", "leather")},
}
SG_PRON = {"he", "she", "it", "this", "that", "everyone", "everybody", "someone",
           "somebody", "nobody", "nothing", "everything"}
PL_PRON = {"they", "we", "these", "those"}
NEG_WORDS = {"no", "nothing", "nobody", "never", "none", "nowhere", "neither"}
VOWEL_SOUND_EXC = {"hour", "honest", "honor", "heir"}
CONS_SOUND_EXC = {"university", "user", "useful", "unit", "unique", "european", "one", "united", "uniform"}


def third_person(lemma: str) -> str:
    irregular = {"be": "is", "have": "has", "do": "does", "go": "goes"}
    if lemma in irregular:
        return irregular[lemma]
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lemma + "es"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"


# ---------- Rule registry ----------
@dataclass
class Hit:
    tokens: list
    fix: str
    why: str


RULES: List[tuple] = []


def rule(rid: str, category: str):
    def deco(fn: Callable) -> Callable:
        RULES.append((rid, category, fn))
        return fn
    return deco


@rule("SVA", "Agreement")
def subject_verb_agreement(sent) -> Iterator[Hit]:
    for t in sent:
        if t.dep_ != "nsubj" or t.tag_ == "WDT" or t.head.pos_ not in ("VERB", "AUX"):
            continue
        v, low, vl = t.head, t.lower_, t.head.lower_
        compound = any(c.dep_ == "conj" for c in t.children) and any(c.dep_ == "cc" for c in t.children)
        sg = not compound and (low in SG_PRON or t.tag_ in ("NN", "NNP"))
        pl = compound or low in PL_PRON or t.tag_ in ("NNS", "NNPS")
        if sg:
            if v.tag_ == "VBP" or vl in ("are", "were"):
                fix = {"are": "is", "were": "was"}.get(vl, third_person(v.lemma_))
                yield Hit([t, v], f"{t.text} {fix}", "A third-person singular subject requires the singular verb form (-s/-es, is/was).")
            for a in v.children:
                if a.dep_ == "aux" and a.lower_ in ("do", "have") and a.tag_ == "VBP":
                    yield Hit([t, a], f"{t.text} {third_person(a.lemma_)}", "Third-person singular subjects take does/has, not do/have.")
        elif pl and (v.tag_ == "VBZ" or vl in ("is", "was")):
            fix = {"is": "are", "was": "were"}.get(vl, v.lemma_)
            yield Hit([t, v], f"{t.text} {fix}", "A plural or coordinated subject requires the plural verb form.")


@rule("PREP", "Collocation")
def preposition_selection(sent) -> Iterator[Hit]:
    for t in sent:
        lemma = t.lemma_.lower()
        for c in t.children:
            if c.dep_ != "prep" or c.i != t.i + 1:
                continue
            if lemma in ALLOWED_PREPS and c.lower_ in PREPS and c.lower_ not in ALLOWED_PREPS[lemma]:
                yield Hit([t, c], f"{t.text} {PREFERRED_PREP[lemma]}",
                          f"'{lemma}' selects the preposition '{PREFERRED_PREP[lemma]}'.")
            if lemma in EXTRA_PREP and c.lower_ == EXTRA_PREP[lemma]:
                yield Hit([t, c], t.text, f"'{lemma}' is transitive; the preposition '{c.text}' is redundant.")


@rule("TENSE", "Tense")
def tense_clash(sent) -> Iterator[Hit]:
    anchor = None
    toks = list(sent)
    for i, t in enumerate(toks):
        if t.lower_ in PAST_WORDS:
            anchor = t
        elif t.lower_ == "last" and i + 1 < len(toks) and toks[i + 1].lower_ in LAST_UNITS:
            anchor = t
        elif t.lower_ == "in" and i + 1 < len(toks) and toks[i + 1].text.isdigit() and len(toks[i + 1].text) == 4 \
                and int(toks[i + 1].text) < time.localtime().tm_year:
            anchor = toks[i + 1]
        if anchor:
            break
    if not anchor:
        return
    for t in toks:
        if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj") and t.tag_ in ("VBP", "VBZ"):
            yield Hit([t, anchor], f"past form of '{t.lemma_}'",
                      f"The past-time expression '{anchor.text}' requires simple past (VBD), not present.")


@rule("DOSUP", "Auxiliary")
def auxiliary_complementation(sent) -> Iterator[Hit]:
    for t in sent:
        if t.dep_ != "aux":
            continue
        head = t.head
        if t.lemma_ == "do" and head.pos_ == "VERB" and head.tag_ in ("VBZ", "VBD", "VBN", "VBG"):
            yield Hit([t, head], f"{t.text} {head.lemma_}", "After do-support the lexical verb must be a bare infinitive.")
        elif t.tag_ == "MD" and head.pos_ == "VERB" and head.tag_ in ("VBZ", "VBD", "VBG"):
            yield Hit([t, head], f"{t.text} {head.lemma_}", "A modal verb must be followed by the bare infinitive.")


@rule("NEG", "Polarity")
def double_negation(sent) -> Iterator[Hit]:
    negs = [t for t in sent if t.dep_ == "neg" or t.lower_ in NEG_WORDS]
    if len(negs) > 1 and "only" not in sent.text.lower():
        yield Hit(negs, "use one negator with any-/ever-",
                  "Standard English allows one clausal negation; use a negative polarity item (anything, anybody, ever).")


@rule("COUNT", "Countability")
def countability(sent) -> Iterator[Hit]:
    for t in sent:
        stem = t.lower_[:-1] if t.lower_.endswith("s") else t.lower_
        if t.tag_ == "NNS" and stem in MASS:
            yield Hit([t], stem, f"'{stem}' is a mass noun and has no plural.")
        if t.lower_ in MASS or stem in MASS:
            for c in t.children:
                if c.lower_ in PLURAL_QUANT and c.dep_ in ("amod", "det", "quantmod"):
                    yield Hit([c, t], f"much / a lot of {stem}",
                              f"'{c.text}' combines only with countable plurals.")
        if t.tag_ == "NNS" and stem not in MASS:
            for c in t.children:
                if c.lower_ in MASS_QUANT and c.dep_ in ("amod", "det", "quantmod"):
                    yield Hit([c, t], f"many {t.text}", "'much' combines only with uncountable nouns.")


@rule("ADJ", "Noun phrase")
def adjective_order(sent) -> Iterator[Hit]:
    for t in sent:
        if t.pos_ != "NOUN":
            continue
        adjs = sorted([c for c in t.children if c.dep_ == "amod" and c.lower_ in ADJ_RANK], key=lambda x: x.i)
        for a, b in zip(adjs, adjs[1:]):
            if ADJ_RANK[a.lower_] > ADJ_RANK[b.lower_]:
                yield Hit([a, b, t], f"{b.text} {a.text} {t.text}",
                          "Prenominal order: size → age → colour → material.")


@rule("ART", "Determiner")
def a_an(sent) -> Iterator[Hit]:
    toks = list(sent)
    for a, b in zip(toks, toks[1:]):
        if a.lower_ not in ("a", "an") or not b.is_alpha:
            continue
        w = b.lower_
        vowel = (w[0] in "aeiou" and w not in CONS_SOUND_EXC) or w in VOWEL_SOUND_EXC
        if a.lower_ == "a" and vowel:
            yield Hit([a, b], f"an {b.text}", "Use 'an' before a vowel sound.")
        elif a.lower_ == "an" and not vowel:
            yield Hit([a, b], f"a {b.text}", "Use 'a' before a consonant sound.")


DURATION_NOUNS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
                  "minute", "minutes", "decade", "decades"}
NUM_EXEMPT = {"percent", "hundred", "thousand", "million", "billion", "dozen", "o'clock", "kg", "km", "cm"}


@rule("PERF", "Tense")
def perfect_with_past_adverb(sent) -> Iterator[Hit]:
    toks = list(sent)
    anchor = next((t for t in toks if t.lower_ in PAST_WORDS), None)
    if anchor is None:
        anchor = next((a for a, b in zip(toks, toks[1:]) if a.lower_ == "last" and b.lower_ in LAST_UNITS), None)
    if anchor is None:
        return
    for t in toks:
        if t.tag_ == "VBN" and any(c.dep_ == "aux" and c.lemma_ == "have" for c in t.children):
            aux = next(c for c in t.children if c.dep_ == "aux" and c.lemma_ == "have")
            yield Hit([aux, t, anchor], f"simple past of '{t.lemma_}'",
                      f"Present perfect cannot combine with the definite past time expression '{anchor.text}'.")


@rule("DCOMP", "Comparison")
def double_comparative(sent) -> Iterator[Hit]:
    toks = list(sent)
    for a, b in zip(toks, toks[1:]):
        if a.lower_ in ("more", "most") and b.tag_ in ("JJR", "JJS") and b.lower_ not in ("more", "most", "less"):
            yield Hit([a, b], b.text, f"'{b.text}' is already comparative/superlative; do not add '{a.text}'.")


@rule("SINCEFOR", "Preposition")
def since_for(sent) -> Iterator[Hit]:
    toks = list(sent)
    for i, t in enumerate(toks):
        if t.lower_ == "since":
            window = toks[i + 1:i + 3]
            if any(w.lower_ in DURATION_NOUNS for w in window) and not any(w.lower_ in ("then", "last") for w in window):
                yield Hit([t] + window[:2], "for + duration", "'since' marks a starting point; durations take 'for'.")
        elif t.lower_ == "for" and i + 1 < len(toks) and toks[i + 1].text.isdigit() and len(toks[i + 1].text) == 4:
            yield Hit([t, toks[i + 1]], "since + year", "A starting point (a year) takes 'since', not 'for'.")


@rule("NUMPL", "Nominal morphology")
def number_plural(sent) -> Iterator[Hit]:
    for t in sent:
        if t.tag_ != "NN" or t.lower_ in NUM_EXEMPT or t.lower_ in MASS:
            continue
        if t.i > 0 and t.doc[t.i - 1].text == "-":
            continue  # compound modifier: "a two-year plan"
        for c in t.children:
            if c.dep_ == "nummod" and c.like_num and c.lower_ not in ("one", "1", "zero") and "." not in c.text:
                plural = t.text[:-1] + "ies" if t.text.endswith("y") and t.text[-2:-1] not in "aeiou" else t.text + "s"
                yield Hit([c, t], f"{c.text} {plural}", "Nouns after numerals greater than one take the plural.")


# ---------- Engine ----------
def analyze(text: str):
    t0 = time.perf_counter()
    doc = load_nlp()(text)
    report = Report(text=text, sentence_count=0, issue_count=0)
    report.trace.append(f"Parsed {len(doc)} tokens with en_core_web_sm (Universal Dependencies).")
    for si, sent in enumerate(doc.sents):
        report.sentence_count += 1
        report.trace.append(f"Sentence {si + 1}: \"{sent.text.strip()}\"")
        for rid, cat, fn in RULES:
            hits = list(fn(sent))
            report.trace.append(f"  {rid:<6} -> {len(hits)} hit(s)")
            for h in hits:
                report.issues.append(Issue(
                    rule_id=rid, category=cat, sentence_index=si,
                    token_indices=sorted({x.i for x in h.tokens}),
                    focus=" … ".join(x.text for x in sorted(h.tokens, key=lambda x: x.i)),
                    fix=h.fix, explanation=h.why, l1_note=L1_NOTES.get(rid)))
    report.issue_count = len(report.issues)
    report.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return report, doc


def highlighted(doc, issues: List[Issue]) -> str:
    flagged = {i for iss in issues for i in iss.token_indices}
    out = []
    for t in doc:
        w = html.escape(t.text)
        out.append(f"<mark>{w}</mark>{t.whitespace_}" if t.i in flagged else f"{w}{t.whitespace_}")
    return "".join(out)


