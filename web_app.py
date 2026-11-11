import html
import re
import time
from typing import List, Optional

import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field
from spacy import displacy

st.set_page_config(
    page_title="L2-SyntaxLens",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap');"
)

GLOBAL_CSS = FONT_IMPORT + """
:root {
    --bg: #0a0c11;
    --panel: #10131a;
    --panel-2: #141821;
    --border: #1e2430;
    --border-2: #2a3140;
    --text: #e6e9ef;
    --text-2: #9aa3b2;
    --text-3: #667085;
    --accent: #818cf8;
    --rose: #f87171;
    --emerald: #34d399;
    --ui: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
}
html, body, .stApp, .stMarkdown, p, label, li, span, div { font-family: var(--ui); }
.stApp { background: var(--bg); color: var(--text-2); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1180px; }
h1, h2, h3, h4 { font-family: var(--ui) !important; color: var(--text); letter-spacing: -0.01em; }
code, pre { font-family: var(--mono) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.sb-label { font-size: .68rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
            color: var(--text-3); margin: 18px 0 8px 0; }
details.sb-group { border: 1px solid var(--border); border-radius: 8px; padding: 0 12px; margin-bottom: 8px;
                   background: var(--panel-2); }
details.sb-group > summary { cursor: pointer; list-style: none; padding: 9px 0; font-size: .82rem;
                             font-weight: 500; color: var(--text); display: flex; justify-content: space-between; }
details.sb-group > summary::-webkit-details-marker { display: none; }
details.sb-group > summary::after { content: "+"; color: var(--text-3); }
details.sb-group[open] > summary::after { content: "–"; }
.pill-wrap { display: flex; flex-wrap: wrap; gap: 6px; padding: 2px 0 12px 0; }
.pill { display: inline-block; font-family: var(--mono); font-size: .68rem; padding: 2px 9px;
        border-radius: 999px; border: 1px solid var(--border-2); color: var(--text-2); white-space: nowrap; }
.pill.indigo  { color: #a5b4fc; border-color: #312e81; background: rgba(99,102,241,.10); }
.pill.cyan    { color: #67e8f9; border-color: #164e63; background: rgba(34,211,238,.08); }
.pill.rose    { color: #fca5a5; border-color: #7f1d1d; background: rgba(248,113,113,.08); }
.pill.emerald { color: #6ee7b7; border-color: #065f46; background: rgba(52,211,153,.08); }
.pill.amber   { color: #fcd34d; border-color: #78350f; background: rgba(251,191,36,.08); }
.pill.violet  { color: #c4b5fd; border-color: #4c1d95; background: rgba(167,139,250,.08); }

/* Header */
.hdr { display: flex; justify-content: space-between; align-items: center; padding: 18px 22px;
       background: var(--panel); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 20px;
       box-shadow: 0 1px 2px rgba(0,0,0,.4); }
.hdr-left { display: flex; align-items: center; gap: 12px; }
.hdr-title { margin: 0; font-size: 1.35rem; font-weight: 700; color: var(--text); }
.hdr-right { text-align: right; }
.hdr-right .lbl { font-size: .62rem; text-transform: uppercase; letter-spacing: .1em; color: var(--text-3); }
.hdr-right .val { font-family: var(--mono); font-size: 1.1rem; color: var(--text); }

/* Input */
div[data-testid="stTextInput"] label p, div[data-testid="stSelectbox"] label p {
    font-size: .68rem !important; font-weight: 600 !important; letter-spacing: .1em;
    text-transform: uppercase; color: var(--text-3) !important; }
div[data-testid="stTextInput"] input {
    font-family: var(--mono) !important; background: var(--panel) !important; color: var(--text);
    border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 11px 13px !important; }
div[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; box-shadow: none !important; }
div[data-baseweb="select"] > div { background: var(--panel-2) !important; border: 1px solid var(--border) !important;
                                   border-radius: 8px !important; }

/* Stat bar */
.statbar { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--border);
           border-radius: 10px; background: var(--panel); margin: 16px 0 20px 0; overflow: hidden; }
.stat { padding: 14px 18px; border-right: 1px solid var(--border); }
.stat:last-child { border-right: none; }
.stat .lbl { font-size: .62rem; text-transform: uppercase; letter-spacing: .1em; color: var(--text-3); }
.stat .val { font-family: var(--mono); font-size: 1.25rem; color: var(--text); margin-top: 3px; }
.stat .val.ok { color: var(--emerald); } .stat .val.bad { color: var(--rose); }
@media (max-width: 800px) { .statbar { grid-template-columns: repeat(2, 1fr); } }

/* Tabs */
div[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
button[data-baseweb="tab"] { background: transparent !important; padding: 8px 14px !important; height: auto !important; }
button[data-baseweb="tab"] p { font-size: .82rem !important; font-weight: 500 !important; color: var(--text-3) !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--text) !important; }
div[data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px !important; }
div[data-baseweb="tab-border"] { display: none !important; }

/* Diagnostic cards */
.utter { font-family: var(--mono); color: var(--text); padding: 10px 14px; border: 1px solid var(--border);
         border-radius: 8px; background: var(--panel); margin: 16px 0 14px 0; font-size: .95rem; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px;
        margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.35); }
.card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; gap: 10px; }
.card-title { margin: 0; font-size: 1rem; font-weight: 600; color: var(--text); }
.chip-row { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 14px; align-items: flex-end; }
.chip-col { display: flex; flex-direction: column; gap: 5px; }
.chip-lbl { font-size: .6rem; text-transform: uppercase; letter-spacing: .1em; color: var(--text-3); }
.chip { font-family: var(--mono); font-size: .86rem; padding: 6px 11px; border-radius: 6px; word-break: break-word; }
.chip.focal  { color: #fecaca; background: rgba(248,113,113,.10); border: 1px solid rgba(248,113,113,.35); }
.chip.target { color: #a7f3d0; background: rgba(52,211,153,.10);  border: 1px solid rgba(52,211,153,.35); }
.arrow { color: var(--text-3); padding-bottom: 7px; }
.rationale { margin: 0; padding: 2px 0 2px 12px; border-left: 2px solid var(--accent);
             color: var(--text-2); font-style: italic; font-size: .88rem; line-height: 1.55; }
.empty { padding: 18px 20px; border: 1px solid var(--border); border-radius: 10px; background: var(--panel);
         color: var(--emerald); font-size: .92rem; }

/* Trace */
.trace { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px 4px;
         font-family: var(--mono); font-size: .8rem; }
.trace-line { display: flex; gap: 12px; padding: 4px 14px; color: var(--text-2); }
.trace-line .n { color: var(--text-3); }
.trace-line.viol { color: #fca5a5; }
.trace-line.done { color: #6ee7b7; }

div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
div[data-testid="stCode"] pre { border: 1px solid var(--border); border-radius: 10px; background: var(--bg) !important; }
div[data-testid="stDownloadButton"] button { background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border-2); border-radius: 8px; font-weight: 500; }
div[data-testid="stDownloadButton"] button:hover { border-color: var(--accent); color: var(--text); }
"""

st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)

DEP_TREE_CSS = FONT_IMPORT + """
html, body { margin: 0; padding: 0; background: transparent; }
.dep-tree-container { background: #090d16; border: 1px solid #1e2430; border-radius: 10px; padding: 16px 12px;
                      overflow-x: auto; overflow-y: hidden; box-sizing: border-box; }
.dep-tree-container::-webkit-scrollbar { height: 8px; }
.dep-tree-container::-webkit-scrollbar-thumb { background: #2a3140; border-radius: 8px; }
.dep-tree-container svg { background: transparent !important; display: block; margin: 0 auto; }
.dep-tree-container text { font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace !important; }
.dep-tree-container .displacy-word  { fill: #f1f5f9 !important; font-size: 15px; font-weight: 500; }
.dep-tree-container .displacy-tag   { fill: #22d3ee !important; font-size: 11px; }
.dep-tree-container .displacy-label { fill: #c7d2fe !important; font-size: 11px; }
.dep-tree-container .displacy-arrow path { stroke: #818cf8 !important; stroke-width: 1.5; fill: none; }
.dep-tree-container .displacy-arrow path:last-child { fill: #22d3ee !important; stroke: #22d3ee !important; stroke-width: 1; }
.dep-tree-container .displacy-arrow textPath { fill: #c7d2fe !important; }
.dep-tree-container .displacy-arrow:hover path { stroke: #22d3ee !important; stroke-width: 2.2; }
.dep-tree-container svg:hover .displacy-arrow:not(:hover) { opacity: .5; }
"""


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


nlp = load_nlp()


class DiagnosticError(BaseModel):
    category: str
    error_type: str
    token_focus: str
    suggested_correction: str
    linguistic_rule: str


class SyntaxDiagnosticReport(BaseModel):
    raw_utterance: str
    status: str
    error_count: int = 0
    diagnostics: List[DiagnosticError] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)
    pedagogical_feedback: str = ""


VERB_PREP_COLLOCATIONS = {
    "depend": "on", "rely": "on", "insist": "on", "listen": "to", "interested": "in",
    "good": "at", "wait": "for", "graduate": "from", "agree": "with", "consist": "of",
}

PAST_TIME_ADVERBIALS = [
    "yesterday", "ago", "last", "formerly", "previously",
    "in 2020", "in 2021", "in 2022", "in 2023", "in 2024",
]

PLURAL_DETERMINERS = ["these", "those", "many", "several", "few"]
SINGULAR_DETERMINERS = ["this", "that", "each", "every", "a", "an"]

UNCOUNTABLE_NOUNS = {
    "information", "advice", "furniture", "equipment", "homework", "knowledge",
    "luggage", "baggage", "evidence", "progress", "traffic", "news", "software",
    "vocabulary", "feedback", "money", "music", "weather", "research", "luck",
    "work", "damage", "garbage", "rubbish", "machinery", "scenery", "mail",
}
COUNT_QUANTIFIERS = {
    "many": "much", "few": "little", "several": "some", "numerous": "much",
    "a few": "a little", "fewer": "less", "these": "this", "those": "that",
}
MASS_NOUN_PLURAL_EXCEPTIONS = {"works", "researches", "moneys", "musics"}

ADJ_ORDER_CLASSES = [
    ("opinion", {"beautiful", "lovely", "nice", "good", "bad", "terrible", "wonderful", "ugly",
                 "interesting", "boring", "awful", "great", "excellent", "pretty", "delicious"}),
    ("size", {"big", "small", "large", "tiny", "huge", "little", "tall", "short", "long", "enormous", "massive"}),
    ("age", {"old", "new", "young", "ancient", "modern", "recent", "antique"}),
    ("shape", {"round", "square", "flat", "triangular", "oval", "rectangular", "circular"}),
    ("color", {"red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey", "pink",
               "purple", "orange", "golden", "silver"}),
    ("origin", {"french", "german", "turkish", "english", "american", "chinese", "japanese", "italian",
                "spanish", "british", "russian", "greek", "indian"}),
    ("material", {"wooden", "plastic", "metal", "cotton", "silk", "leather", "glass", "stone",
                  "golden", "woolen", "paper", "steel"}),
    ("purpose", {"cooking", "running", "sleeping", "walking", "writing", "sewing"}),
]
ADJ_RANK = {}
for _rank, (_cls, _words) in enumerate(ADJ_ORDER_CLASSES):
    for _w in _words:
        ADJ_RANK.setdefault(_w, (_rank, _cls))

YEAR_PAST_RE = re.compile(r"\bin\s+(19\d{2}|20(?:[01]\d|2[0-4]))\b", re.I)


def _mass_lemma(token) -> Optional[str]:
    """Return the mass-noun base if the token is a (possibly pluralized) mass noun."""
    low = token.text.lower()
    lemma = token.lemma_.lower()
    if low in MASS_NOUN_PLURAL_EXCEPTIONS:
        return None
    if lemma in UNCOUNTABLE_NOUNS:
        return lemma
    if low.endswith("s") and low[:-1] in UNCOUNTABLE_NOUNS and low != "news":
        return low[:-1]
    if low in UNCOUNTABLE_NOUNS:
        return low
    return None


def diagnose_l2_syntax(text: str):
    start_time = time.perf_counter()
    doc = nlp(text)
    report = SyntaxDiagnosticReport(raw_utterance=text, status="GRAMMATICAL")
    trace = report.execution_trace
    seen = set()

    def flag(category, error_type, focus, correction, rule, trace_msg=None):
        key = (error_type, focus, correction)
        if key in seen:
            return
        seen.add(key)
        report.diagnostics.append(DiagnosticError(
            category=category, error_type=error_type, token_focus=focus,
            suggested_correction=correction, linguistic_rule=rule,
        ))
        if trace_msg:
            trace.append(trace_msg)

    trace.append(f"Segmented utterance into {len(doc)} tokens.")

    # 1. Subject-Verb Agreement (SVA)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ in ["VERB", "AUX"]:
            verb = token.head
            trace.append(f"Checking agreement: nsubj('{token.text}') -> head('{verb.text}')")
            is_3sg = (
                token.lower_ in ["he", "she", "it"]
                or (token.tag_ in ["NN", "NNP"] and token.text.lower() not in ["i", "you"])
            )
            if is_3sg and verb.tag_ == "VBP":
                lem = verb.lemma_.lower()
                if lem in ("go", "do"):
                    third = verb.lemma_ + "es"
                elif lem == "have":
                    third = "has"
                elif lem.endswith(("s", "sh", "ch", "x", "z")):
                    third = verb.lemma_ + "es"
                elif lem.endswith("y") and lem[-2:-1] not in "aeiou":
                    third = verb.lemma_[:-1] + "ies"
                else:
                    third = verb.lemma_ + "s"
                flag(
                    "Morphosyntactic Agreement", "Subject-Verb Concord Error",
                    token.text + " " + verb.text, token.text + " " + third,
                    "Third-person singular subjects select for 3sg present inflection (-s, XPOS: VBZ).",
                    "Violation: SVA clash between 3sg subject and uninflected verb.",
                )

    # 2. Prepositional Collocations / Subcategorization frames
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"]:
            lemma = token.lemma_.lower()
            if lemma in VERB_PREP_COLLOCATIONS:
                target_prep = VERB_PREP_COLLOCATIONS[lemma]
                for child in token.children:
                    if child.dep_ in ["prep", "adp"] and child.text.lower() != target_prep:
                        flag(
                            "Lexicosyntactic Collocation", "Erroneous Preposition Selection",
                            token.text + " " + child.text, token.text + " " + target_prep,
                            f"Predicate '{lemma}' selects preposition '{target_prep}' via complementation.",
                            f"Violation: '{lemma}' subcategorized with '{child.text}' instead of '{target_prep}'.",
                        )

    # 3. Tense-Aspect Concordance
    past_anchor = None
    for token in doc:
        if token.dep_ in ["advmod", "pobj", "npadvmod", "amod"] or token.lower_ in ("yesterday", "ago"):
            if token.lower_ in [a for a in PAST_TIME_ADVERBIALS if " " not in a]:
                past_anchor = token.text
                break
    if past_anchor is None:
        m = YEAR_PAST_RE.search(text)
        if m:
            past_anchor = m.group(0)

    if past_anchor:
        trace.append(f"Detected past deictic anchor: '{past_anchor}'.")
        anchor_tok = next((t for t in doc if t.text == past_anchor or t.text == past_anchor.split()[-1]), None)
        governing = None
        if anchor_tok is not None:
            for anc in anchor_tok.ancestors:
                if anc.pos_ in ("VERB", "AUX"):
                    governing = anc
                    break
        for token in doc:
            if governing is not None and token is not governing:
                continue  # only the clause hosting the past adverbial is constrained
            if token.pos_ == "VERB" and token.tag_ in ["VBP", "VBZ"]:
                flag(
                    "Temporal Syntax", "Tense-Aspect Clash",
                    token.text + " ... " + past_anchor,
                    "Inflect '" + token.text + "' into past tense (VBD)",
                    f"Past deictic anchor '{past_anchor}' requires simple past morphology.",
                    f"Violation: Present verb '{token.text}' clashed with past anchor '{past_anchor}'.",
                )

    # 4. Double Marking / Auxiliary Complementation
    for token in doc:
        if token.dep_ == "aux" and token.lemma_ == "do":
            main_verb = token.head
            if main_verb.pos_ == "VERB" and main_verb.tag_ in ["VBZ", "VBD"]:
                flag(
                    "Auxiliary Complementation", "Double Marking / Auxiliary Redundancy",
                    token.text + " " + main_verb.text, token.text + " " + main_verb.lemma_,
                    "Do-support absorbs inflection; lexical verb must be bare infinitive (VB).",
                    "Violation: Double marking detected on both auxiliary and main verb.",
                )
            elif main_verb.pos_ == "VERB" and main_verb.tag_ == "VBP" and token.tag_ in ("VBZ", "VBD") \
                    and main_verb.lower_ != main_verb.lemma_.lower():
                flag(
                    "Auxiliary Complementation", "Double Marking / Auxiliary Redundancy",
                    token.text + " " + main_verb.text, token.text + " " + main_verb.lemma_,
                    "Do-support absorbs inflection; lexical verb must be bare infinitive (VB).",
                    "Violation: Double marking detected on both auxiliary and main verb.",
                )

    # 5. Determiner-Noun Number Concordance
    for token in doc:
        if token.dep_ == "det":
            det_text = token.text.lower()
            head_noun = token.head
            if head_noun.pos_ == "NOUN":
                if det_text in PLURAL_DETERMINERS and head_noun.tag_ == "NN" \
                        and _mass_lemma(head_noun) is None:
                    if det_text in ["these", "those"]:
                        flag(
                            "NP Morphosyntax", "Demonstrative Concord Clash",
                            token.text + " " + head_noun.text,
                            ("this" if det_text == "these" else "that") + " " + head_noun.text
                            + "  |  " + token.text + " " + head_noun.lemma_ + "s",
                            "Proximal/distal plural demonstratives require plural head complement.",
                            "Violation: Plural demonstrative attached to singular head noun.",
                        )
                    else:
                        flag(
                            "NP Morphosyntax", "Determiner-Noun Number Mismatch",
                            token.text + " " + head_noun.text, token.text + " " + head_noun.lemma_ + "s",
                            f"Plural determiner '{det_text}' requires plural head noun morphology (NNS).",
                            "Violation: Plural determiner attached to singular count noun.",
                        )
                elif det_text in SINGULAR_DETERMINERS and head_noun.tag_ == "NNS" \
                        and _mass_lemma(head_noun) is None:
                    nxt = doc[token.i + 1].lower_ if token.i + 1 < len(doc) else ""
                    if nxt not in ("few", "couple", "lot", "number", "bit"):
                        flag(
                            "NP Morphosyntax", "Determiner-Noun Number Mismatch",
                            token.text + " " + head_noun.text, token.text + " " + head_noun.lemma_,
                            f"Singular determiner '{det_text}' requires singular head noun morphology (NN).",
                            "Violation: Singular determiner attached to plural noun.",
                        )

    # 6. Non-Standard Double Negation
    neg_tokens = [t for t in doc if t.dep_ == "neg" or t.text.lower() in
                  ["no", "never", "not", "n't", "nothing", "nobody", "nowhere", "none"]]
    neg_tokens = list(dict.fromkeys(neg_tokens))
    if len(neg_tokens) > 1:
        flag(
            "Negative Polarity", "Non-Standard Double Negation",
            " ... ".join(t.text for t in neg_tokens),
            "Use single negation with standard negative polarity item (e.g., anything / anybody)",
            "Standard English syntax enforces single clausal negation or NPI licensing.",
            "Violation: Multiple negative operators detected within local clausal scope.",
        )

    # 7. Modal Auxiliary Complementation
    for token in doc:
        if token.pos_ == "AUX" and token.tag_ == "MD":
            main_verb = token.head
            for child in token.children:
                if child.text.lower() == "to":
                    flag(
                        "Modal Syntax", "Infinitival Marker with Modal",
                        token.text + " to " + main_verb.text, token.text + " " + main_verb.text,
                        "Modal auxiliaries select bare infinitival complements without 'to'.",
                        "Violation: Modal auxiliary co-occurs with infinitival 'to'.",
                    )
            if main_verb.tag_ in ["VBZ", "VBD", "VBG"] and main_verb is not token:
                flag(
                    "Modal Syntax", "Inflected Modal Complement",
                    token.text + " " + main_verb.text, token.text + " " + main_verb.lemma_,
                    "Modal auxiliaries strictly govern bare infinitive predicates (VB).",
                    "Violation: Modal governs an inflected lexical verb.",
                )

    # 8. Uncountable / Mass Noun Pluralization & Quantifier Clashes
    for token in doc:
        base = _mass_lemma(token)
        if base is None or token.pos_ not in ("NOUN", "PROPN"):
            continue
        is_pluralized = token.text.lower() != base
        quant = None
        for child in token.children:
            if child.lower_ in COUNT_QUANTIFIERS and child.dep_ in ("amod", "det", "quantmod", "nummod", "advmod"):
                quant = child
                break
        if quant is None and token.i > 0 and doc[token.i - 1].lower_ in ("many", "several", "numerous", "fewer"):
            quant = doc[token.i - 1]
        if quant is not None:
            repair = COUNT_QUANTIFIERS[quant.lower_]
            flag(
                "Mass Noun Syntax",
                "Mass Noun Pluralization + Quantifier Clash" if is_pluralized else "Mass Noun Quantifier Clash",
                quant.text + " " + token.text, f"{repair} {base}",
                f"'{base}' is a non-count (mass) noun: it resists plural -s and takes mass quantifiers (much/little/some), not count quantifiers.",
                f"Violation: Count quantifier '{quant.text}' combined with mass noun '{base}'.",
            )
        elif is_pluralized:
            flag(
                "Mass Noun Syntax", "Mass Noun Pluralization", token.text, base,
                f"'{base}' is a non-count (mass) noun and lacks a plural form; use the singular, or a partitive (a piece of {base}).",
                f"Violation: Mass noun '{base}' realized with plural morphology '{token.text}'.",
            )
        else:
            for child in token.children:
                if child.dep_ == "det" and child.lower_ in ("a", "an"):
                    flag(
                        "Mass Noun Syntax", "Indefinite Article with Mass Noun",
                        child.text + " " + token.text, f"a piece of {base}",
                        f"Non-count noun '{base}' is incompatible with the indefinite article; use a partitive construction.",
                        f"Violation: Indefinite article '{child.text}' attached to mass noun '{base}'.",
                    )

    # 9. Prenominal Adjective Ordering Constraints
    for noun in doc:
        if noun.pos_ not in ("NOUN", "PROPN"):
            continue
        adjs = [c for c in noun.children if c.dep_ == "amod" and c.i < noun.i and c.lower_ in ADJ_RANK]
        adjs.sort(key=lambda t: t.i)
        if len(adjs) < 2:
            continue
        ranks = [ADJ_RANK[a.lower_][0] for a in adjs]
        trace.append(
            "Checking adjective order: "
            + " > ".join(f"{a.text}[{ADJ_RANK[a.lower_][1]}]" for a in adjs)
            + f" -> '{noun.text}'"
        )
        if ranks != sorted(ranks):
            ordered = sorted(adjs, key=lambda a: (ADJ_RANK[a.lower_][0], a.i))
            flag(
                "Adjective Ordering", "Prenominal Adjective Order Violation",
                " ".join(a.text for a in adjs) + " " + noun.text,
                " ".join(a.text for a in ordered) + " " + noun.text,
                "Prenominal modifiers follow the canonical order: opinion < size < age < shape < colour < origin < material < purpose.",
                "Violation: Prenominal adjective sequence deviates from canonical semantic ordering.",
            )

    report.error_count = len(report.diagnostics)
    if report.error_count > 0:
        report.status = "ERRORS_DETECTED"
        report.pedagogical_feedback = f"Identified {report.error_count} morphosyntactic violation(s)."
        trace.append(f"Diagnosis completed: {report.error_count} anomaly/anomalies flagged.")
    else:
        report.pedagogical_feedback = "Sentence conforms to modeled standard syntactic rules."
        trace.append("Diagnosis completed: No structural violations found.")

    latency = (time.perf_counter() - start_time) * 1000
    return report, doc, latency


# ── UI helpers ───────────────────────────────────────────────────────────────
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def html_block(markup: str):
    st.markdown("".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


CATEGORY_PILL = {
    "Morphosyntactic Agreement": ("indigo", "Agreement"),
    "Lexicosyntactic Collocation": ("cyan", "Collocation"),
    "Temporal Syntax": ("amber", "Tense-Aspect"),
    "Auxiliary Complementation": ("violet", "Auxiliary"),
    "NP Morphosyntax": ("emerald", "NP Syntax"),
    "Negative Polarity": ("rose", "Polarity"),
    "Modal Syntax": ("violet", "Modal"),
    "Mass Noun Syntax": ("emerald", "Mass Noun"),
    "Adjective Ordering": ("cyan", "Adj Order"),
}

SCOPE_GROUPS = [
    ("Agreement & NP", True, [("indigo", "SVA"), ("emerald", "NP Syntax"),
                              ("emerald", "Mass Nouns"), ("cyan", "Adj Order")]),
    ("Complementation", False, [("cyan", "Collocation"), ("violet", "Auxiliary"), ("violet", "Modal")]),
    ("Tense & Polarity", False, [("amber", "Tense-Aspect"), ("rose", "Negation")]),
]

TEST_CORPORA = {
    "Subject-Verb Agreement": "He go to the university library every afternoon.",
    "Preposition Selection": "Many students depend of their family for tuition.",
    "Tense-Aspect Clash": "She writes an academic essay yesterday.",
    "Auxiliary Double Marking": "The applicant does not has enough credits.",
    "Determiner-Noun Number": "These book contains fundamental computational theories.",
    "Double Negation": "The student didn't write nothing during the lecture.",
    "Mass Noun Quantifier": "Many informations are available in the library.",
    "Adjective Ordering": "She bought a red big car.",
    "Well-Formed Control": "The student depends on primary sources and wrote an essay yesterday.",
}


def _apply_preset():
    st.session_state["utterance"] = TEST_CORPORA[st.session_state["preset"]]


def render_dependency_tree(doc):
    svg = displacy.render(
        doc, style="dep", jupyter=False,
        options={
            "compact": len(doc) > 12, "distance": 120, "color": "#f1f5f9", "bg": "transparent",
            "font": "JetBrains Mono", "arrow_stroke": 2, "arrow_width": 8,
            "word_spacing": 42, "offset_x": 40,
        },
    )
    m = re.search(r'height="(\d+(?:\.\d+)?)"', svg)
    svg_h = int(float(m.group(1))) if m else 380
    page = f"<style>{DEP_TREE_CSS}</style><div class=\"dep-tree-container\">{svg}</div>"
    components.html(page, height=min(max(svg_h + 50, 260), 720), scrolling=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
if "utterance" not in st.session_state:
    st.session_state["utterance"] = TEST_CORPORA["Subject-Verb Agreement"]

with st.sidebar:
    html_block('<div class="sb-label">Linguistic Scope</div>')
    for title, is_open, pills in SCOPE_GROUPS:
        pill_html = "".join(f'<span class="pill {c}">{esc(n)}</span>' for c, n in pills)
        html_block(
            f'<details class="sb-group"{" open" if is_open else ""}><summary>{esc(title)}</summary>'
            f'<div class="pill-wrap">{pill_html}</div></details>'
        )
    html_block('<div class="sb-label">Benchmark Corpora</div>')
    st.selectbox("Preset", list(TEST_CORPORA.keys()), key="preset", on_change=_apply_preset,
                 label_visibility="collapsed")

# ── Main ─────────────────────────────────────────────────────────────────────
utterance = st.session_state.get("utterance", "")
report = doc = None
latency = 0.0
if utterance and utterance.strip():
    report, doc, latency = diagnose_l2_syntax(utterance)

lat_display = f"{latency:.1f} ms" if report else "—"
html_block(f"""
    <div class="hdr">
      <div class="hdr-left">
        <h1 class="hdr-title">L2-SyntaxLens</h1>
        <span class="pill indigo">v2.5 Laboratory Workbench</span>
      </div>
      <div class="hdr-right"><div class="lbl">Latency</div><div class="val">{lat_display}</div></div>
    </div>
""")

st.text_input("Learner sentence", key="utterance", placeholder="Type a sentence…")

if report is not None:
    ok = report.status == "GRAMMATICAL"
    html_block(f"""
        <div class="statbar">
          <div class="stat"><div class="lbl">Tokens</div><div class="val">{len(doc)}</div></div>
          <div class="stat"><div class="lbl">Latency</div><div class="val">{latency:.1f} ms</div></div>
          <div class="stat"><div class="lbl">Violations</div><div class="val {'ok' if ok else 'bad'}">{report.error_count}</div></div>
          <div class="stat"><div class="lbl">Status</div><div class="val {'ok' if ok else 'bad'}">{'Well-formed' if ok else 'Flagged'}</div></div>
        </div>
    """)

    tab_diag, tab_syntax, tab_tokens, tab_trace, tab_schema = st.tabs([
        "Diagnostic Overview",
        "Universal Dependencies Tree (SVG)",
        "Morphosyntactic Token Matrix (DataFrame)",
        "Deterministic Rule Trace",
        "Export Pydantic Schema (JSON)",
    ])

    with tab_diag:
        html_block(f'<div class="utter">{esc(report.raw_utterance)}</div>')
        if report.diagnostics:
            for diag in report.diagnostics:
                pill_cls, pill_label = CATEGORY_PILL.get(diag.category, ("indigo", diag.category))
                html_block(f"""
                    <div class="card">
                      <div class="card-head">
                        <h3 class="card-title">{esc(diag.error_type)}</h3>
                        <span class="pill {pill_cls}" title="{esc(diag.category)}">{esc(pill_label)}</span>
                      </div>
                      <div class="chip-row">
                        <div class="chip-col"><span class="chip-lbl">Focal Violation</span>
                          <span class="chip focal">{esc(diag.token_focus)}</span></div>
                        <span class="arrow">→</span>
                        <div class="chip-col"><span class="chip-lbl">Target Reconstruction</span>
                          <span class="chip target">{esc(diag.suggested_correction)}</span></div>
                      </div>
                      <p class="rationale">{esc(diag.linguistic_rule)}</p>
                    </div>
                """)
        else:
            html_block('<div class="empty">No violations detected.</div>')

    with tab_syntax:
        render_dependency_tree(doc)

    with tab_tokens:
        token_records = [{
            "Idx": t.i,
            "Token": t.text,
            "Lemma": t.lemma_,
            "UPOS": t.pos_,
            "XPOS": t.tag_,
            "Dep": t.dep_,
            "Head": t.head.text,
            "Morphology": str(t.morph) or "—",
        } for t in doc]
        df_kwargs = dict(hide_index=True)
        try:
            st.dataframe(pd.DataFrame(token_records), width="stretch", **df_kwargs)
        except TypeError:  # older Streamlit releases
            st.dataframe(pd.DataFrame(token_records), use_container_width=True, **df_kwargs)

    with tab_trace:
        lines = []
        for i, step in enumerate(report.execution_trace):
            cls = "viol" if step.startswith("Violation") else \
                  "done" if step.startswith("Diagnosis completed: No") else ""
            lines.append(f'<div class="trace-line {cls}"><span class="n">{i + 1:02d}</span><span>{esc(step)}</span></div>')
        html_block(f'<div class="trace">{"".join(lines)}</div>')

    with tab_schema:
        json_payload = report.model_dump_json(indent=2)
        st.download_button("Download JSON", data=json_payload,
                           file_name="l2_syntaxlens_report.json", mime="application/json")
        st.code(json_payload, language="json")
else:
    html_block('<div class="empty" style="color:var(--text-2)">Enter a sentence to begin.</div>')
