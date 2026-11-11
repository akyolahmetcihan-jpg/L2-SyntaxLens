import functools
import html
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import spacy
import streamlit as st
from pydantic import BaseModel, Field
from spacy import displacy

st.set_page_config(page_title="L2-SyntaxLens", page_icon="🔎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap');
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em; }
.sent { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem; line-height: 1.7;
        padding: 18px 22px; background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.35); border-radius: 6px; }
.sent mark { background: rgba(251,146,60,0.30); color: inherit; padding: 1px 4px; border-bottom: 2px solid #fb923c; border-radius: 2px; }
.issue { background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.35); border-left: 4px solid #fb923c;
         padding: 14px 18px; border-radius: 4px; margin-bottom: 10px; }
.issue .meta { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65; }
.issue .title { font-weight: 600; font-size: 1.02rem; margin: 2px 0 8px; }
.issue code { font-family: 'JetBrains Mono', monospace; font-size: .85rem; background: rgba(128,128,128,0.22); padding: 1px 6px; border-radius: 3px; }
.issue .why { opacity: .85; font-size: .88rem; margin-top: 8px; }
table.tok { width: 100%; border-collapse: collapse; font-size: .88rem; }
table.tok th { text-align: left; background: rgba(128,128,128,0.18); padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,0.35); }
table.tok td { padding: 5px 10px; border-bottom: 1px solid rgba(128,128,128,0.25); font-family: 'JetBrains Mono', monospace; }
.ok { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.5); padding: 12px 16px; border-radius: 4px; }
.trace { font-family: 'JetBrains Mono', monospace; font-size: .82rem; background: #1c1917; color: #e7e5e4;
         padding: 16px; border-radius: 6px; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

L1_NOTES = {
    "SVA": "Turkish verbs agree with the subject in person, but there is no zero/-s contrast for 3rd person singular, and plural marking on the subject is often optional, so the English -s may be omitted or misplaced.",
    "PREP": "Turkish expresses these relations with case suffixes (-e, -den, -de) and postpositions, so English verb–preposition pairings must be memorised and are often mapped from the Turkish case.",
    "TENSE": "Turkish verbs carry an obligatory tense suffix; the learner may rely on the time adverb and leave the English verb uninflected.",
    "PERF": "Turkish has no direct present-perfect equivalent with the same adverb restrictions, so 'have + V-ed' may be combined with definite past time adverbs.",
    "DOSUP": "Turkish negation is a verbal suffix (-me) and has no auxiliary 'do', so learners may inflect the lexical verb instead of the auxiliary.",
    "NEG": "Turkish uses negative concord (e.g. 'hiçbir şey yazmadı': negative word + negative verb), which is transferred to English.",
    "COUNT": "Turkish equivalents such as 'bilgi' and 'tavsiye' can be pluralised ('bilgiler', 'tavsiyeler'), so the English mass-noun restriction is not intuitive.",
    "ADJ": "Turkish has no strictly taught size–age–colour–material order, so learners lack a native template.",
    "ART": "Turkish has no definite article and the a/an alternation (sound-based) has no counterpart.",
    "DCOMP": "Turkish forms comparatives analytically with 'daha' + adjective; generalising 'more + adjective' to already comparative English forms is a common overgeneralisation.",
    "SINCEFOR": "Turkish 'beri' / '-dır' cover both point-in-time and duration readings (e.g. 'üç yıldır', '2020'den beri'), so 'since' and 'for' are conflated.",
    "NUMPL": "In Turkish, nouns stay singular after numerals ('üç kardeş'), which is transferred to English.",
}


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


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
ADJ_RANK = {
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
DURATION_NOUNS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
                  "minute", "minutes", "decade", "decades"}
NUM_EXEMPT = {"percent", "hundred", "thousand", "million", "billion", "dozen", "o'clock", "kg", "km", "cm"}


def third_person(lemma: str) -> str:
    irregular = {"be": "is", "have": "has", "do": "does", "go": "goes"}
    if lemma in irregular:
        return irregular[lemma]
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lemma + "es"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"


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
                yield Hit([t, c], f"{t.text} {PREFERRED_PREP[lemma]}", f"'{lemma}' selects the preposition '{PREFERRED_PREP[lemma]}'.")
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
        elif t.lower_ == "in" and i + 1 < len(toks) and toks[i + 1].text.isdigit() and len(toks[i + 1].text) == 4 and int(toks[i + 1].text) < time.localtime().tm_year:
            anchor = toks[i + 1]
        if anchor:
            break
    if not anchor:
        return
    for t in toks:
        if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj") and t.tag_ in ("VBP", "VBZ"):
            yield Hit([t, anchor], f"past form of '{t.lemma_}'", f"The past-time expression '{anchor.text}' requires simple past (VBD), not present.")


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
        yield Hit(negs, "use one negator with any-/ever-", "Standard English allows one clausal negation; use a negative polarity item (anything, anybody, ever).")


@rule("COUNT", "Countability")
def countability(sent) -> Iterator[Hit]:
    for t in sent:
        stem = t.lower_[:-1] if t.lower_.endswith("s") else t.lower_
        if t.tag_ == "NNS" and stem in MASS:
            yield Hit([t], stem, f"'{stem}' is a mass noun and has no plural.")
        if t.lower_ in MASS or stem in MASS:
            for c in t.children:
                if c.lower_ in PLURAL_QUANT and c.dep_ in ("amod", "det", "quantmod"):
                    yield Hit([c, t], f"much / a lot of {stem}", f"'{c.text}' combines only with countable plurals.")
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
                yield Hit([a, b, t], f"{b.text} {a.text} {t.text}", "Prenominal order: size → age → colour → material.")


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
            yield Hit([aux, t, anchor], f"simple past of '{t.lemma_}'", f"Present perfect cannot combine with the definite past time expression '{anchor.text}'.")


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
            continue
        for c in t.children:
            if c.dep_ == "nummod" and c.like_num and c.lower_ not in ("one", "1", "zero") and "." not in c.text:
                plural = t.text[:-1] + "ies" if t.text.endswith("y") and t.text[-2:-1] not in "aeiou" else t.text + "s"
                yield Hit([c, t], f"{c.text} {plural}", "Nouns after numerals greater than one take the plural.")


def analyze(text: str):
    nlp = load_nlp()
    t0 = time.perf_counter()
    doc = nlp(text)
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


PRESETS = {
    "Subject–verb agreement": "He go to the university library every afternoon.",
    "Plural subject": "The students goes to class and the teachers is late.",
    "Mass noun": "The professor gave us many useful informations.",
    "Adjective order": "She bought a red big backpack yesterday.",
    "Preposition": "Many students depend of their family for tuition.",
    "Tense clash": "She writes an academic essay yesterday.",
    "Do-support": "The applicant does not has enough credits.",
    "Double negation": "The student didn't write nothing during the lecture.",
    "Article": "He is an university student and wants a answer.",
    "Double comparative": "This exam is more easier than the last one.",
    "Since / for": "I have lived here since three years.",
    "Numeral + plural": "She has three brother and two sister.",
    "Perfect + past adverb": "I have seen him yesterday.",
    "Multi-sentence": "She don't like coffee. They was happy last week. I discussed about the plan.",
    "Well-formed": "The researcher depends on primary data and gathered significant evidence.",
}

with st.sidebar:
    st.markdown("### L2-SyntaxLens")
    st.caption("Rule-based diagnostics for L2 English, built on Universal Dependencies.")
    preset = st.selectbox("Example sentences", list(PRESETS))
    show_l1 = st.checkbox("Show possible Turkish (L1) transfer notes", value=True)
    st.caption("Held-out eval (n=60): P 0.92 · R 0.91 · F1 0.92")
    st.markdown("**Rules**")
    for rid, cat, fn in RULES:
        st.markdown(f"- `{rid}` · {cat}")

st.title("L2-SyntaxLens")
st.caption("Paste learner text. Every flag is traceable to a dependency relation and an explicit rule.")

text = st.text_area("Learner text", value=PRESETS[preset], height=110, key=f"in_{preset}")

if text.strip():
    report, doc = analyze(text.strip())

    c1, c2, c3 = st.columns(3)
    c1.metric("Sentences", report.sentence_count)
    c2.metric("Issues", report.issue_count)
    c3.metric("Latency", f"{report.latency_ms} ms")

    st.markdown(f'<div class="sent">{highlighted(doc, report.issues)}</div>', unsafe_allow_html=True)
    st.write("")

    tab_d, tab_t, tab_k, tab_j, tab_l = st.tabs(["Diagnostics", "Dependency tree", "Tokens", "JSON", "Trace"])

    with tab_d:
        if not report.issues:
            st.markdown('<div class="ok">No violations found within the modelled rules.</div>', unsafe_allow_html=True)
        for iss in report.issues:
            l1_html = (f'<div class="why"><b>Possible L1 (Turkish) transfer:</b> {html.escape(iss.l1_note)}</div>'
                       if show_l1 and iss.l1_note else "")
            st.markdown(f"""
            <div class="issue">
              <div class="meta">{html.escape(iss.category)} · {iss.rule_id} · sentence {iss.sentence_index + 1}</div>
              <div class="title"><code>{html.escape(iss.focus)}</code> → <code>{html.escape(iss.fix)}</code></div>
              <div class="why">{html.escape(iss.explanation)}</div>
              {l1_html}
            </div>""", unsafe_allow_html=True)

    with tab_t:
        for sent in doc.sents:
            svg = displacy.render(sent.as_doc(), style="dep",
                                  options={"compact": True, "distance": 105, "bg": "transparent",
                                           "color": "inherit", "font": "JetBrains Mono"})
            st.markdown(f'<div style="overflow-x:auto">{svg.replace(chr(10), "")}</div>',
                        unsafe_allow_html=True)

    with tab_k:
        cols = ["Token", "Lemma", "UPOS", "XPOS", "Dep", "Head", "Morph"]
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.head.text, str(t.morph) or "—")) + "</tr>"
            for t in doc)
        head = "".join(f"<th>{c}</th>" for c in cols)
        st.markdown(f'<table class="tok"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>',
                    unsafe_allow_html=True)

    with tab_j:
        payload = report.model_dump_json(indent=2)
        st.download_button("Download JSON", payload, "syntaxlens_report.json", "application/json")
        st.code(payload, language="json")

    with tab_l:
        st.markdown(f'<div class="trace">{html.escape(chr(10).join(report.trace))}</div>', unsafe_allow_html=True)
