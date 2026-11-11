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

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="L2-SyntaxLens | Diagnostic Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;500;600&display=swap');"
)

GLOBAL_CSS = (
    FONT_IMPORT
    + """
:root {
    --bg-0: #05070d;
    --bg-1: #090d16;
    --bg-2: #0f1524;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --indigo-950: #1e1b4b;
    --indigo-500: #6366f1;
    --indigo-400: #818cf8;
    --indigo-300: #a5b4fc;
    --cyan-400: #22d3ee;
    --rose-400: #fb7185;
    --rose-950: #4c0519;
    --emerald-400: #34d399;
    --emerald-950: #022c22;
    --amber-400: #fbbf24;
    --text-0: #f1f5f9;
    --text-1: #cbd5e1;
    --text-2: #94a3b8;
    --text-3: #64748b;
    --ui: 'Plus Jakarta Sans', 'Inter', -apple-system, 'Segoe UI', sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, 'Cascadia Code', Consolas, monospace;
}

html, body, [class*="css"], .stApp, .stMarkdown, p, label, li, span, div {
    font-family: var(--ui);
}
.stApp {
    background:
        radial-gradient(1100px 520px at 12% -8%, rgba(99,102,241,.16), transparent 60%),
        radial-gradient(900px 480px at 100% 0%, rgba(34,211,238,.09), transparent 55%),
        var(--bg-0);
    color: var(--text-1);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }

h1, h2, h3, h4, h5 { font-family: var(--ui) !important; letter-spacing: -0.02em; color: var(--text-0); }
code, pre, kbd { font-family: var(--mono) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,21,36,.96), rgba(9,13,22,.98));
    border-right: 1px solid var(--slate-800);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.sb-brand { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.sb-logo {
    width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, var(--indigo-500), var(--cyan-400)); font-size:18px;
    box-shadow: 0 6px 18px rgba(99,102,241,.35);
}
.sb-title { font-weight:800; font-size:1.05rem; color:var(--text-0); line-height:1.1; }
.sb-sub { font-family:var(--mono); font-size:.66rem; color:var(--text-3); letter-spacing:.04em; }
.sb-section {
    margin: 14px 0 6px 0; font-size:.68rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:var(--text-3);
}
details.sb-group {
    background: rgba(30,41,59,.35); border:1px solid var(--slate-800); border-radius:12px;
    padding:2px 10px; margin-bottom:8px; transition: border-color .2s ease;
}
details.sb-group[open] { border-color: var(--indigo-950); background: rgba(30,27,75,.22); }
details.sb-group:hover { border-color: var(--slate-700); }
details.sb-group > summary {
    cursor:pointer; list-style:none; padding:8px 2px; font-size:.82rem; font-weight:600; color:var(--text-0);
    display:flex; align-items:center; justify-content:space-between;
}
details.sb-group > summary::-webkit-details-marker { display:none; }
details.sb-group > summary::after { content:"＋"; color:var(--text-3); font-size:.8rem; }
details.sb-group[open] > summary::after { content:"－"; color:var(--indigo-300); }
.sb-card {
    background: rgba(9,13,22,.7); border:1px solid var(--slate-800); border-radius:10px;
    padding:8px 10px; margin: 0 0 8px 0;
}
.sb-card .k { display:flex; gap:6px; align-items:center; margin-bottom:3px; }
.sb-card .d { font-size:.74rem; color:var(--text-2); line-height:1.4; }
.pill {
    display:inline-flex; align-items:center; gap:5px; font-family:var(--mono); font-size:.66rem; font-weight:600;
    padding:2px 8px; border-radius:999px; border:1px solid var(--slate-700); color:var(--text-1);
    background: rgba(51,65,85,.35); letter-spacing:.02em; white-space:nowrap;
}
.pill.indigo { color:var(--indigo-300); border-color:#3730a3; background:rgba(55,48,163,.22); }
.pill.cyan   { color:var(--cyan-400);   border-color:#155e75; background:rgba(8,145,178,.15); }
.pill.rose   { color:#fda4af;           border-color:#9f1239; background:rgba(159,18,57,.22); }
.pill.emerald{ color:#6ee7b7;           border-color:#047857; background:rgba(4,120,87,.2); }
.pill.amber  { color:#fcd34d;           border-color:#92400e; background:rgba(146,64,14,.22); }
.pill.violet { color:#c4b5fd;           border-color:#5b21b6; background:rgba(91,33,182,.22); }
.pill.slate  { color:var(--text-1); }

/* ── Hero ── */
.hero {
    position:relative; overflow:hidden; border-radius:20px; padding:26px 30px; margin-bottom:16px;
    background: linear-gradient(135deg, rgba(30,27,75,.55), rgba(15,21,36,.7) 55%, rgba(8,47,73,.45));
    border:1px solid var(--indigo-950);
    box-shadow: 0 20px 50px -20px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.05);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.hero::before {
    content:""; position:absolute; inset:-1px; pointer-events:none; border-radius:20px;
    background: radial-gradient(500px 160px at 0% 0%, rgba(129,140,248,.18), transparent 70%);
}
.hero-top { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; flex-wrap:wrap; position:relative; }
.hero-title {
    margin:0; font-size:2.35rem; font-weight:800; letter-spacing:-0.035em; line-height:1.05;
    background: linear-gradient(90deg, #fff 0%, var(--indigo-300) 55%, var(--cyan-400) 100%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero-badges { display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }
.hero-sub { margin:10px 0 0 0; max-width:640px; color:var(--text-2); font-size:.95rem; line-height:1.55; }
.latency-box {
    min-width:170px; text-align:right; padding:12px 16px; border-radius:14px;
    background: rgba(5,7,13,.55); border:1px solid var(--slate-800);
}
.latency-box .lbl { font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--text-3); font-weight:700; }
.latency-box .val { font-family:var(--mono); font-size:1.7rem; font-weight:600; color:var(--cyan-400); line-height:1.2; }
.latency-box .val small { font-size:.8rem; color:var(--text-2); margin-left:3px; }
.pulse {
    display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--emerald-400);
    box-shadow:0 0 0 0 rgba(52,211,153,.6); animation:pulse 2s infinite; margin-right:6px;
}
@keyframes pulse { 70% { box-shadow:0 0 0 7px rgba(52,211,153,0);} 100% { box-shadow:0 0 0 0 rgba(52,211,153,0);} }

/* ── Input ── */
div[data-testid="stTextInput"] label p, div[data-testid="stSelectbox"] label p {
    font-size:.7rem !important; font-weight:700 !important; letter-spacing:.12em; text-transform:uppercase; color:var(--text-3) !important;
}
div[data-testid="stTextInput"] input {
    font-family:var(--mono) !important; font-size:1rem; color:var(--text-0);
    background: rgba(15,21,36,.75) !important; border:1px solid var(--slate-800) !important; border-radius:12px !important;
    padding:12px 14px !important; box-shadow: inset 0 1px 2px rgba(0,0,0,.4);
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--indigo-500) !important; box-shadow: 0 0 0 3px rgba(99,102,241,.22) !important;
}
div[data-baseweb="select"] > div {
    background: rgba(15,21,36,.75) !important; border:1px solid var(--slate-800) !important; border-radius:10px !important;
}

/* ── Stat bar ── */
.statbar { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin: 14px 0 18px 0; }
.stat {
    background: rgba(15,21,36,.7); border:1px solid var(--slate-800); border-radius:14px; padding:14px 16px;
    box-shadow: 0 8px 24px -14px rgba(0,0,0,.9); backdrop-filter: blur(8px);
}
.stat .lbl { font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--text-3); font-weight:700; }
.stat .val { font-family:var(--mono); font-size:1.5rem; font-weight:600; color:var(--text-0); margin-top:2px; }
.stat .val.ok { color:var(--emerald-400); } .stat .val.bad { color:var(--rose-400); }
.stat .val.cy { color:var(--cyan-400); } .stat .val.in { color:var(--indigo-300); }
.stat .sub { font-size:.72rem; color:var(--text-2); margin-top:2px; }
@media (max-width: 900px) { .statbar { grid-template-columns: repeat(2, 1fr);} }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    gap:6px; background: rgba(15,21,36,.6); border:1px solid var(--slate-800); border-radius:14px; padding:5px;
}
button[data-baseweb="tab"] {
    background:transparent !important; border-radius:10px !important; padding:8px 14px !important; height:auto !important;
}
button[data-baseweb="tab"] p { font-size:.8rem !important; font-weight:600 !important; color:var(--text-2) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79,70,229,.35), rgba(8,145,178,.2)) !important;
    box-shadow: inset 0 0 0 1px rgba(129,140,248,.35);
}
button[data-baseweb="tab"][aria-selected="true"] p { color:var(--text-0) !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none !important; }

/* ── Panels ── */
.panel-title { display:flex; align-items:center; gap:10px; margin:18px 0 12px 0; }
.panel-title h4 { margin:0; font-size:1.02rem; font-weight:700; }
.panel-title .line { flex:1; height:1px; background: linear-gradient(90deg, var(--slate-800), transparent); }

.diag-card {
    position:relative; border-radius:16px; padding:20px 22px; margin-bottom:14px;
    background: linear-gradient(160deg, rgba(30,27,75,.35), rgba(15,21,36,.75));
    border:1px solid var(--indigo-950); box-shadow: 0 16px 36px -22px rgba(0,0,0,.95), inset 0 1px 0 rgba(255,255,255,.04);
    backdrop-filter: blur(10px); transition: transform .18s ease, border-color .18s ease;
}
.diag-card:hover { transform: translateY(-1px); border-color:#3730a3; }
.diag-head { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
.diag-title { display:flex; align-items:center; gap:10px; margin:0; font-size:1.12rem; font-weight:800; color:#fff; letter-spacing:-0.01em; }
.diag-title .idx {
    font-family:var(--mono); font-size:.7rem; color:var(--rose-400); background:var(--rose-950);
    border:1px solid #9f1239; border-radius:8px; padding:2px 7px;
}
.chip-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
.chip-col { display:flex; flex-direction:column; gap:5px; min-width:0; }
.chip-lbl { font-size:.6rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; color:var(--text-3); }
.chip {
    font-family:var(--mono); font-size:.88rem; font-weight:500; padding:7px 12px; border-radius:10px; word-break:break-word;
}
.chip.focal { color:#fecdd3; background:rgba(244,63,94,.12); border:1px solid rgba(244,63,94,.45); box-shadow:0 0 18px -8px rgba(244,63,94,.6); }
.chip.target { color:#a7f3d0; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.45); box-shadow:0 0 18px -8px rgba(16,185,129,.6); }
.arrow { align-self:flex-end; padding-bottom:8px; color:var(--text-3); font-family:var(--mono); }
.rationale {
    margin:0; padding:10px 14px; border-left:3px solid var(--indigo-500); border-radius:0 10px 10px 0;
    background: rgba(99,102,241,.07); color:var(--text-1); font-style:italic; font-size:.9rem; line-height:1.55;
}
.rationale b { font-style:normal; color:var(--indigo-300); font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; display:block; margin-bottom:2px; }

.ok-card {
    border-radius:16px; padding:22px 24px; background: linear-gradient(160deg, rgba(2,44,34,.55), rgba(15,21,36,.7));
    border:1px solid rgba(16,185,129,.4); display:flex; gap:16px; align-items:center;
}
.ok-card .ic { font-size:1.8rem; }
.ok-card h4 { margin:0 0 3px 0; color:#6ee7b7; font-size:1.05rem; }
.ok-card p { margin:0; color:var(--text-1); font-size:.9rem; }

.utter-strip {
    font-family:var(--mono); font-size:1rem; color:var(--text-0); padding:12px 16px; border-radius:12px; margin-bottom:12px;
    background: rgba(9,13,22,.8); border:1px dashed var(--slate-700);
}

/* ── Trace ── */
.trace {
    background: var(--bg-1); border:1px solid var(--slate-800); border-radius:14px; padding:14px 6px;
    font-family:var(--mono); font-size:.82rem; box-shadow: inset 0 2px 10px rgba(0,0,0,.45);
}
.trace-line { display:flex; gap:12px; padding:5px 14px; border-radius:8px; color:#7dd3fc; }
.trace-line:hover { background: rgba(30,41,59,.5); }
.trace-line .n { color:var(--text-3); user-select:none; }
.trace-line.viol { color:#fda4af; background: rgba(244,63,94,.06); }
.trace-line.done { color:#6ee7b7; }
.trace-line.chk  { color:var(--indigo-300); }

/* ── DataFrame / JSON / code ── */
div[data-testid="stDataFrame"] { border:1px solid var(--slate-800); border-radius:14px; overflow:hidden; }
div[data-testid="stCode"] pre, div[data-testid="stJson"] {
    border:1px solid var(--slate-800); border-radius:14px; background: var(--bg-1) !important;
}
div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #4f46e5, #0891b2); color:#fff; border:none; border-radius:10px;
    font-weight:600; padding:8px 18px; box-shadow: 0 8px 20px -8px rgba(79,70,229,.7);
}
div[data-testid="stDownloadButton"] button:hover { filter:brightness(1.1); color:#fff; }

.dep-shell-title { color: var(--text-2); font-size:.8rem; margin-bottom:8px; }
"""
)

st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)

# Styling injected into the isolated iframe holding the displaCy SVG.
DEP_TREE_CSS = (
    FONT_IMPORT
    + """
html, body { margin:0; padding:0; background:transparent; }
.dep-tree-container {
    background:#090d16;
    background-image: radial-gradient(600px 220px at 50% 0%, rgba(99,102,241,.10), transparent 70%);
    border:1px solid #1e1b4b; border-radius:16px; padding:18px 14px;
    box-shadow: 0 20px 40px -24px rgba(0,0,0,.95), inset 0 1px 0 rgba(255,255,255,.04);
    overflow-x:auto; overflow-y:hidden; box-sizing:border-box;
}
.dep-tree-container::-webkit-scrollbar { height:8px; }
.dep-tree-container::-webkit-scrollbar-thumb { background:#334155; border-radius:8px; }
.dep-tree-container svg { background:transparent !important; display:block; margin:0 auto; }
.dep-tree-container text { font-family:'JetBrains Mono', ui-monospace, Consolas, monospace !important; }
.dep-tree-container .displacy-word { fill:#f8fafc !important; font-weight:600; font-size:15px; }
.dep-tree-container .displacy-tag  { fill:#22d3ee !important; font-size:11px; letter-spacing:.04em; }
.dep-tree-container .displacy-label { fill:#c7d2fe !important; font-size:11px; font-weight:500; }
.dep-tree-container .displacy-arrow path { stroke:#818cf8 !important; stroke-width:1.6; fill:none; }
.dep-tree-container .displacy-arrow path:last-child { fill:#22d3ee !important; stroke:#22d3ee !important; stroke-width:1; }
.dep-tree-container .displacy-arrow textPath { fill:#c7d2fe !important; }
.dep-tree-container .displacy-arrow:hover path { stroke:#22d3ee !important; stroke-width:2.4; }
.dep-tree-container .displacy-arrow:hover textPath { fill:#fff !important; }
.dep-tree-container .displacy-arrow { transition: opacity .2s; cursor:pointer; }
.dep-tree-container svg:hover .displacy-arrow:not(:hover) { opacity:.45; }
"""
)

# ─────────────────────────────────────────────────────────────────────────────
# NLP resources (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


nlp = load_nlp()

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic v2 schemas
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic resources
# ─────────────────────────────────────────────────────────────────────────────
VERB_PREP_COLLOCATIONS = {
    "depend": "on",
    "rely": "on",
    "insist": "on",
    "listen": "to",
    "interested": "in",
    "good": "at",
    "wait": "for",
    "graduate": "from",
    "agree": "with",
    "consist": "of",
}

PAST_TIME_ADVERBIALS = [
    "yesterday", "ago", "last", "formerly", "previously",
    "in 2020", "in 2021", "in 2022", "in 2023", "in 2024",
]

PLURAL_DETERMINERS = ["these", "those", "many", "several", "few"]
SINGULAR_DETERMINERS = ["this", "that", "each", "every", "a", "an"]

# Mass (uncountable) nouns and their quantifier repairs.
UNCOUNTABLE_NOUNS = {
    "information", "advice", "furniture", "equipment", "homework", "knowledge",
    "luggage", "baggage", "evidence", "progress", "traffic", "news", "software",
    "vocabulary", "feedback", "money", "music", "weather", "research", "luck",
    "work", "damage", "garbage", "rubbish", "machinery", "scenery", "mail",
}
# Count-only quantifiers that clash with mass nouns -> mass counterpart.
COUNT_QUANTIFIERS = {
    "many": "much", "few": "little", "several": "some", "numerous": "much",
    "a few": "a little", "fewer": "less", "these": "this", "those": "that",
}
# Lexemes whose plural -s is attested and legitimate (avoid false positives).
MASS_NOUN_PLURAL_EXCEPTIONS = {"works", "researches", "moneys", "musics"}

# Prenominal adjective ordering (Quirk / Dixon-style semantic classes).
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
        # earlier class wins for ambiguous lexemes (e.g. "golden")
        ADJ_RANK.setdefault(_w, (_rank, _cls))

YEAR_PAST_RE = re.compile(r"\bin\s+(19\d{2}|20(?:[01]\d|2[0-4]))\b", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic engine
# ─────────────────────────────────────────────────────────────────────────────
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
            category=category,
            error_type=error_type,
            token_focus=focus,
            suggested_correction=correction,
            linguistic_rule=rule,
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
                flag(
                    "Morphosyntactic Agreement",
                    "Subject-Verb Concord Error",
                    token.text + " " + verb.text,
                    token.text + " " + (
                        verb.lemma_ + "es" if verb.lemma_.lower() in ("go", "do")
                        else "has" if verb.lemma_.lower() == "have"
                        else verb.lemma_ + "es" if verb.lemma_.lower().endswith(("s", "sh", "ch", "x", "z"))
                        else verb.lemma_[:-1] + "ies" if verb.lemma_.lower().endswith("y") and verb.lemma_[-2:-1].lower() not in "aeiou"
                        else verb.lemma_ + "s"
                    ),
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
                            "Lexicosyntactic Collocation",
                            "Erroneous Preposition Selection",
                            token.text + " " + child.text,
                            token.text + " " + target_prep,
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
                continue  # only the clause that hosts the past adverbial is constrained
            if token.pos_ == "VERB" and token.tag_ in ["VBP", "VBZ"]:
                flag(
                    "Temporal Syntax",
                    "Tense-Aspect Clash",
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
                    "Auxiliary Complementation",
                    "Double Marking / Auxiliary Redundancy",
                    token.text + " " + main_verb.text,
                    token.text + " " + main_verb.lemma_,
                    "Do-support absorbs inflection; lexical verb must be bare infinitive (VB).",
                    "Violation: Double marking detected on both auxiliary and main verb.",
                )
            elif main_verb.pos_ == "VERB" and main_verb.tag_ == "VBP" and token.tag_ in ("VBZ", "VBD") \
                    and main_verb.lower_ != main_verb.lemma_.lower():
                flag(
                    "Auxiliary Complementation",
                    "Double Marking / Auxiliary Redundancy",
                    token.text + " " + main_verb.text,
                    token.text + " " + main_verb.lemma_,
                    "Do-support absorbs inflection; lexical verb must be bare infinitive (VB).",
                    "Violation: Double marking detected on both auxiliary and main verb.",
                )

    # 5. Determiner-Noun Number Concordance (NP agreement)
    for token in doc:
        if token.dep_ == "det":
            det_text = token.text.lower()
            head_noun = token.head
            if head_noun.pos_ == "NOUN":
                if det_text in PLURAL_DETERMINERS and head_noun.tag_ == "NN" \
                        and _mass_lemma(head_noun) is None:
                    if det_text in ["these", "those"]:
                        flag(
                            "NP Morphosyntax",
                            "Demonstrative Concord Clash",
                            token.text + " " + head_noun.text,
                            ("this" if det_text == "these" else "that") + " " + head_noun.text
                            + "  |  " + token.text + " " + head_noun.lemma_ + "s",
                            "Proximal/distal plural demonstratives require plural head complement.",
                            "Violation: Plural demonstrative attached to singular head noun.",
                        )
                    else:
                        flag(
                            "NP Morphosyntax",
                            "Determiner-Noun Number Mismatch",
                            token.text + " " + head_noun.text,
                            token.text + " " + head_noun.lemma_ + "s",
                            f"Plural determiner '{det_text}' requires plural head noun morphology (NNS).",
                            "Violation: Plural determiner attached to singular count noun.",
                        )
                elif det_text in SINGULAR_DETERMINERS and head_noun.tag_ == "NNS" \
                        and _mass_lemma(head_noun) is None:
                    nxt = doc[token.i + 1].lower_ if token.i + 1 < len(doc) else ""
                    if nxt not in ("few", "couple", "lot", "number", "bit"):
                        flag(
                            "NP Morphosyntax",
                            "Determiner-Noun Number Mismatch",
                            token.text + " " + head_noun.text,
                            token.text + " " + head_noun.lemma_,
                            f"Singular determiner '{det_text}' requires singular head noun morphology (NN).",
                            "Violation: Singular determiner attached to plural noun.",
                        )

    # 6. Non-Standard Double Negation
    neg_tokens = [t for t in doc if t.dep_ == "neg" or t.text.lower() in ["no", "never", "not", "n't", "nothing", "nobody", "nowhere", "none"]]
    neg_tokens = list(dict.fromkeys(neg_tokens))
    if len(neg_tokens) > 1:
        focus = " ... ".join(t.text for t in neg_tokens)
        flag(
            "Negative Polarity",
            "Non-Standard Double Negation",
            focus,
            "Use single negation with standard negative polarity item (e.g., anything / anybody)",
            "Standard English syntax enforces single clausal negation or NPI licensing.",
            "Violation: Multiple negative operators detected within local clausal scope.",
        )

    # 7. Modal Auxiliary Complementation (can to go / must goes)
    for token in doc:
        if token.pos_ == "AUX" and token.tag_ == "MD":
            main_verb = token.head
            for child in token.children:
                if child.text.lower() == "to":
                    flag(
                        "Modal Syntax",
                        "Infinitival Marker with Modal",
                        token.text + " to " + main_verb.text,
                        token.text + " " + main_verb.text,
                        "Modal auxiliaries select bare infinitival complements without 'to'.",
                        "Violation: Modal auxiliary co-occurs with infinitival 'to'.",
                    )
            if main_verb.tag_ in ["VBZ", "VBD", "VBG"] and main_verb is not token:
                flag(
                    "Modal Syntax",
                    "Inflected Modal Complement",
                    token.text + " " + main_verb.text,
                    token.text + " " + main_verb.lemma_,
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
                "Mass Noun Quantifier Clash" if not is_pluralized else "Mass Noun Pluralization + Quantifier Clash",
                quant.text + " " + token.text,
                f"{repair} {base}",
                f"'{base}' is a non-count (mass) noun: it resists plural -s and takes mass quantifiers (much/little/some), not count quantifiers.",
                f"Violation: Count quantifier '{quant.text}' combined with mass noun '{base}'.",
            )
        elif is_pluralized:
            flag(
                "Mass Noun Syntax",
                "Mass Noun Pluralization",
                token.text,
                base,
                f"'{base}' is a non-count (mass) noun and lacks a plural form; use the singular, or a partitive (a piece of {base}).",
                f"Violation: Mass noun '{base}' realized with plural morphology '{token.text}'.",
            )
        else:
            for child in token.children:
                if child.dep_ == "det" and child.lower_ in ("a", "an"):
                    flag(
                        "Mass Noun Syntax",
                        "Indefinite Article with Mass Noun",
                        child.text + " " + token.text,
                        f"a piece of {base}",
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
                "Adjective Ordering",
                "Prenominal Adjective Order Violation",
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


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def html_block(markup: str):
    """Render HTML safely through Streamlit markdown (collapsing indentation so it is never a code block)."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


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
    ("Morphosyntax & Agreement", True, [
        ("indigo", "SVA", "3sg concord (VBZ vs VBP) between nsubj and finite head."),
        ("emerald", "NP Syntax", "Determiner–head number concord; mass-noun quantifier clashes."),
        ("cyan", "Adj Order", "Prenominal opinion < size < age < colour < origin < material."),
    ]),
    ("Lexicosyntax & Complementation", False, [
        ("cyan", "Collocation", "Verb/adjective prepositional subcategorization frames."),
        ("violet", "Auxiliary", "Do-support double marking; bare-infinitive modals."),
    ]),
    ("Temporal & Polarity", False, [
        ("amber", "Temporal", "Past deictic adverbial vs. present-tense verb clash."),
        ("rose", "Polarity", "Non-standard double negation / NPI licensing."),
    ]),
]

TEST_CORPORA = {
    "Subject-Verb Agreement Clash": "He go to the university library every afternoon.",
    "Erroneous Preposition Selection": "Many students depend of their family for tuition.",
    "Tense-Aspect Clash with Past Adverb": "She writes an academic essay yesterday.",
    "Double Marking Error (Auxiliary)": "The applicant does not has enough credits.",
    "Determiner-Noun Number Clash": "These book contains fundamental computational theories.",
    "Non-Standard Double Negation": "The student didn't write nothing during the lecture.",
    "Mass Noun Quantifier Clash": "Many informations are available in the library.",
    "Adjective Ordering Violation": "She bought a red big car.",
    "Well-Formed Control Sentence": "The student depends on primary sources and wrote an essay yesterday.",
}

CORPUS_PILLS = {
    "Subject-Verb Agreement Clash": ("indigo", "SVA"),
    "Erroneous Preposition Selection": ("cyan", "COLL"),
    "Tense-Aspect Clash with Past Adverb": ("amber", "TENSE"),
    "Double Marking Error (Auxiliary)": ("violet", "AUX"),
    "Determiner-Noun Number Clash": ("emerald", "NP"),
    "Non-Standard Double Negation": ("rose", "NEG"),
    "Mass Noun Quantifier Clash": ("emerald", "MASS"),
    "Adjective Ordering Violation": ("cyan", "ADJ"),
    "Well-Formed Control Sentence": ("slate", "CTRL"),
}


def _apply_preset():
    st.session_state["utterance"] = TEST_CORPORA[st.session_state["preset"]]


def render_dependency_tree(doc):
    svg = displacy.render(
        doc,
        style="dep",
        jupyter=False,
        options={
            "compact": len(doc) > 12,
            "distance": 120,
            "color": "#f8fafc",
            "bg": "transparent",
            "font": "JetBrains Mono",
            "arrow_stroke": 2,
            "arrow_width": 8,
            "word_spacing": 42,
            "offset_x": 40,
        },
    )
    m = re.search(r'height="(\d+(?:\.\d+)?)"', svg)
    svg_h = int(float(m.group(1))) if m else 380
    page = (
        f"<style>{DEP_TREE_CSS}</style>"
        f'<div class="dep-tree-container">{svg}</div>'
    )
    components.html(page, height=min(max(svg_h + 60, 260), 720), scrolling=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
if "utterance" not in st.session_state:
    st.session_state["utterance"] = TEST_CORPORA["Subject-Verb Agreement Clash"]

with st.sidebar:
    html_block("""
        <div class="sb-brand"><div class="sb-logo">🔬</div>
        <div><div class="sb-title">L2-SyntaxLens</div>
        <div class="sb-sub">UD · spaCy · Pydantic v2</div></div></div>
    """)

    html_block('<div class="sb-section">Linguistic Scope</div>')
    for title, is_open, items in SCOPE_GROUPS:
        cards = "".join(
            f'<div class="sb-card"><div class="k"><span class="pill {c}">{esc(name)}</span></div>'
            f'<div class="d">{esc(desc)}</div></div>'
            for c, name, desc in items
        )
        html_block(
            f'<details class="sb-group"{" open" if is_open else ""}>'
            f"<summary>{esc(title)}</summary>{cards}</details>"
        )

    html_block('<div class="sb-section">Benchmark Corpora Presets</div>')
    st.selectbox(
        "Select Test Utterance",
        list(TEST_CORPORA.keys()),
        key="preset",
        on_change=_apply_preset,
    )
    cur = st.session_state["preset"]
    pc, pn = CORPUS_PILLS[cur]
    html_block(
        f'<div class="sb-card"><div class="k"><span class="pill {pc}">{pn}</span>'
        f'<span class="pill slate">n={len(TEST_CORPORA)} corpora</span></div>'
        f'<div class="d" style="font-family:var(--mono)">{esc(TEST_CORPORA[cur])}</div></div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main — compute first so hero can show latency
# ─────────────────────────────────────────────────────────────────────────────
user_utterance = st.session_state.get("utterance", "")
report = doc = None
latency = 0.0
if user_utterance and user_utterance.strip():
    report, doc, latency = diagnose_l2_syntax(user_utterance)

lat_display = f"{latency:.2f}" if report else "—"
html_block(f"""
    <div class="hero">
      <div class="hero-top">
        <div>
          <div class="hero-badges">
            <span class="pill indigo">v2.5 Laboratory Workbench</span>
            <span class="pill cyan">Universal Dependencies</span>
            <span class="pill slate">en_core_web_sm</span>
          </div>
          <h1 class="hero-title">L2-SyntaxLens</h1>
          <p class="hero-sub">Deterministic morphosyntactic and pedagogical error diagnosis for L2 English learner
          language — dependency-grounded rules, transparent traces, and schema-validated output.</p>
        </div>
        <div class="latency-box">
          <div class="lbl"><span class="pulse"></span>Parse Latency</div>
          <div class="val">{lat_display}<small>ms</small></div>
        </div>
      </div>
    </div>
""")

st.text_input("Input L2 Learner Sentence", key="utterance", placeholder="Type or paste a learner sentence…")

if report is not None:
    ok = report.status == "GRAMMATICAL"
    html_block(f"""
        <div class="statbar">
          <div class="stat"><div class="lbl">Token Count</div><div class="val in">{len(doc)}</div>
            <div class="sub">{sum(1 for t in doc if not t.is_punct)} lexical · {sum(1 for t in doc if t.is_punct)} punct</div></div>
          <div class="stat"><div class="lbl">Parse Latency</div><div class="val cy">{latency:.2f} ms</div>
            <div class="sub">tokenise + parse + rules</div></div>
          <div class="stat"><div class="lbl">Violations</div><div class="val {'ok' if ok else 'bad'}">{report.error_count}</div>
            <div class="sub">{esc(report.pedagogical_feedback)}</div></div>
          <div class="stat"><div class="lbl">Health Status</div>
            <div class="val {'ok' if ok else 'bad'}">{'WELL-FORMED' if ok else 'ANOMALOUS'}</div>
            <div class="sub">{esc(report.status)}</div></div>
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
        html_block(
            '<div class="panel-title"><h4>Pedagogical &amp; Morphosyntactic Diagnostics</h4><div class="line"></div></div>'
            f'<div class="utter-strip">“{esc(report.raw_utterance)}”</div>'
        )
        if report.diagnostics:
            for i, diag in enumerate(report.diagnostics, 1):
                pill_cls, pill_label = CATEGORY_PILL.get(diag.category, ("slate", diag.category))
                html_block(f"""
                    <div class="diag-card">
                      <div class="diag-head">
                        <h3 class="diag-title"><span class="idx">#{i:02d}</span>{esc(diag.error_type)}</h3>
                        <span class="pill {pill_cls}" title="{esc(diag.category)}">{esc(pill_label)}</span>
                      </div>
                      <div class="chip-row">
                        <div class="chip-col"><span class="chip-lbl">Focal Violation</span>
                          <span class="chip focal">{esc(diag.token_focus)}</span></div>
                        <span class="arrow">➔</span>
                        <div class="chip-col"><span class="chip-lbl">Target Reconstruction</span>
                          <span class="chip target">{esc(diag.suggested_correction)}</span></div>
                      </div>
                      <p class="rationale"><b>Linguistic Rationale</b>{esc(diag.linguistic_rule)}</p>
                    </div>
                """)
        else:
            html_block("""
                <div class="ok-card"><div class="ic">✅</div><div>
                <h4>No morphosyntactic violations identified</h4>
                <p>The syntactic parse tree demonstrates grammatical concordance under all modeled rules.</p>
                </div></div>
            """)

    with tab_syntax:
        html_block(
            '<div class="panel-title"><h4>Dependency Parse Graph (Universal Dependencies)</h4><div class="line"></div></div>'
            '<div class="dep-shell-title">Hover an arc to isolate it · scroll horizontally for long sentences.</div>'
        )
        render_dependency_tree(doc)

    with tab_tokens:
        html_block(
            '<div class="panel-title"><h4>Token-Level Morphosyntactic Attributes</h4><div class="line"></div></div>'
        )
        token_records = [{
            "Idx": t.i,
            "Token": t.text,
            "Lemma": t.lemma_,
            "UPOS": t.pos_,
            "XPOS (Penn Treebank)": t.tag_,
            "Dep Rel": t.dep_,
            "Head Token": t.head.text,
            "Morphology": str(t.morph) or "—",
        } for t in doc]
        df_kwargs = dict(
            hide_index=True,
            column_config={
                "Idx": st.column_config.NumberColumn("Idx", width="small"),
                "Token": st.column_config.TextColumn("Token", width="small"),
                "Morphology": st.column_config.TextColumn("Morphology", width="large"),
            },
        )
        try:
            st.dataframe(pd.DataFrame(token_records), width="stretch", **df_kwargs)
        except TypeError:  # older Streamlit releases
            st.dataframe(pd.DataFrame(token_records), use_container_width=True, **df_kwargs)

    with tab_trace:
        html_block(
            '<div class="panel-title"><h4>Step-by-Step Rule Traversal Log</h4><div class="line"></div></div>'
        )
        lines = []
        for i, step in enumerate(report.execution_trace):
            cls = "viol" if step.startswith("Violation") else "done" if step.startswith("Diagnosis completed: No") \
                else "chk" if step.startswith("Checking") or step.startswith("Detected") else ""
            lines.append(
                f'<div class="trace-line {cls}"><span class="n">[{i + 1:02d}]</span><span>{esc(step)}</span></div>'
            )
        html_block(f'<div class="trace">{"".join(lines)}</div>')

    with tab_schema:
        html_block(
            '<div class="panel-title"><h4>Pydantic Structured Diagnostic Schema</h4><div class="line"></div></div>'
        )
        json_payload = report.model_dump_json(indent=2)
        st.download_button(
            "⬇ Download report.json",
            data=json_payload,
            file_name="l2_syntaxlens_report.json",
            mime="application/json",
        )
        st.code(json_payload, language="json")
else:
    html_block("""
        <div class="ok-card" style="border-color:#3730a3;background:rgba(30,27,75,.35)">
        <div class="ic">⌨️</div><div><h4 style="color:#a5b4fc">Awaiting input</h4>
        <p>Enter a learner sentence above or choose a benchmark preset from the sidebar.</p></div></div>
    """)
