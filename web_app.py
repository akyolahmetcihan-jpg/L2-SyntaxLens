import time
from typing import List, Optional
from pydantic import BaseModel, Field
import spacy
from spacy import displacy
import streamlit as st

st.set_page_config(
    page_title="L2-SyntaxLens | Diagnostic Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI / Theme Engine
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #312e81;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    
    .dep-tree-container {
        background-color: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        overflow-x: auto;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.4);
    }
    .dep-tree-container text {
        fill: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .dep-tree-container path, .dep-tree-container line {
        stroke: #6366f1 !important;
        stroke-width: 1.5px !important;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .badge-error {
        background-color: rgba(225, 29, 72, 0.15);
        color: #fb7185;
        border: 1px solid #e11d48;
    }
    .badge-valid {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
    }
    
    .error-card {
        background: linear-gradient(145deg, #13112c 0%, #1a163d 100%);
        border: 1px solid #4338ca;
        border-left: 5px solid #818cf8;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .token-chip {
        background: #1e1b4b;
        color: #f43f5e;
        border: 1px solid #4c1d95;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    .target-chip {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #047857;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .trace-box {
        background-color: #050814;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        line-height: 1.6;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

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
    "depend": "on",
    "rely": "on",
    "insist": "on",
    "listen": "to",
    "interested": "in",
    "good": "at",
    "wait": "for",
    "graduate": "from",
    "agree": "with",
    "consist": "of"
}

PAST_TIME_ADVERBIALS = [
    "yesterday", "ago", "last", "formerly", "previously",
    "in 2020", "in 2021", "in 2022", "in 2023", "in 2024"
]

UNCOUNTABLE_NOUNS = {
    "information": "information",
    "informations": "information",
    "advice": "advice",
    "advices": "advice",
    "research": "research",
    "researches": "research (or studies)",
    "furniture": "furniture",
    "furnitures": "furniture",
    "evidence": "evidence",
    "evidences": "evidence",
    "homework": "homework",
    "homeworks": "homework"
}

PLURAL_QUANTIFIERS = ["many", "several", "few", "these", "those"]
COLOR_WORDS = {"red", "blue", "green", "black", "white", "yellow", "dark"}
SIZE_WORDS = {"big", "large", "small", "huge", "tiny", "tall"}

def get_3sg_present(verb_text: str, lemma: str) -> str:
    irregular = {"be": "is", "have": "has", "do": "does", "go": "goes"}
    if lemma in irregular:
        return irregular[lemma]
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lemma + "es"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"

def diagnose_l2_syntax(text: str):
    start_time = time.perf_counter()
    doc = nlp(text)
    report = SyntaxDiagnosticReport(raw_utterance=text, status="GRAMMATICAL")
    trace = report.execution_trace

    trace.append(f"Tokenized utterance into {len(doc)} nodes via Universal Dependencies.")

    # 1. Subject-Verb Agreement (SVA)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ in ["VERB", "AUX"]:
            verb = token.head
            trace.append(f"Agreement test: nsubj('{token.text}') -> head('{verb.text}')")
            is_3sg = (
                token.lower_ in ["he", "she", "it"] or 
                (token.tag_ in ["NN", "NNP"] and token.text.lower() not in ["i", "you"])
            )
            if is_3sg and verb.tag_ == "VBP":
                target_form = get_3sg_present(verb.text, verb.lemma_)
                report.diagnostics.append(DiagnosticError(
                    category="Morphosyntactic Agreement",
                    error_type="Subject-Verb Concord Error",
                    token_focus=f"{token.text} + {verb.text}",
                    suggested_correction=f"{token.text} {target_form}",
                    linguistic_rule="Third-person singular subjects select for 3sg present inflection (-s/-es, XPOS: VBZ)."
                ))
                trace.append("Violation: SVA clash between 3sg subject and uninflected verb.")

    # 2. Prepositional Collocations
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"]:
            lemma = token.lemma_.lower()
            if lemma in VERB_PREP_COLLOCATIONS:
                target_prep = VERB_PREP_COLLOCATIONS[lemma]
                for child in token.children:
                    if child.dep_ in ["prep", "adp"] and child.text.lower() != target_prep:
                        report.diagnostics.append(DiagnosticError(
                            category="Lexicosyntactic Collocation",
                            error_type="Erroneous Preposition Selection",
                            token_focus=f"{token.text} + {child.text}",
                            suggested_correction=f"{token.text} {target_prep}",
                            linguistic_rule=f"Predicate '{lemma}' selects preposition '{target_prep}' via complementation."
                        ))
                        trace.append(f"Violation: '{lemma}' subcategorized with '{child.text}' instead of '{target_prep}'.")

    # 3. Tense-Aspect Concordance
    past_anchor = None
    for token in doc:
        if token.dep_ in ["advmod", "pobj", "npadvmod"]:
            if any(adv in token.text.lower() for adv in PAST_TIME_ADVERBIALS):
                past_anchor = token.text
                break

    if past_anchor:
        for token in doc:
            if token.pos_ == "VERB" and token.tag_ in ["VBP", "VBZ"]:
                report.diagnostics.append(DiagnosticError(
                    category="Temporal Syntax",
                    error_type="Tense-Aspect Clash",
                    token_focus=f"{token.text} ... {past_anchor}",
                    suggested_correction=f"Use simple past form of '{token.text}' (e.g., {token.lemma_}ed / past irregular)",
                    linguistic_rule=f"Past deictic anchor '{past_anchor}' requires simple past tense morphology (VBD)."
                ))
                trace.append(f"Violation: Finite verb '{token.text}' clashed with past anchor '{past_anchor}'.")

    # 4. Double Marking / Auxiliary Complementation
    for token in doc:
        if token.dep_ == "aux" and token.lemma_ == "do":
            main_verb = token.head
            if main_verb.pos_ == "VERB" and main_verb.tag_ in ["VBZ", "VBD"]:
                target_lemma = "have" if main_verb.lemma_ == "have" else main_verb.lemma_
                report.diagnostics.append(DiagnosticError(
                    category="Auxiliary Complementation",
                    error_type="Double Marking / Auxiliary Redundancy",
                    token_focus=f"{token.text} + {main_verb.text}",
                    suggested_correction=f"{token.text} {target_lemma}",
                    linguistic_rule="Do-support absorbs inflection; lexical verb must be bare infinitive (VB)."
                ))
                trace.append("Violation: Double marking detected on both auxiliary and main verb.")

    # 5. Non-Standard Double Negation
    neg_tokens = [t for t in doc if t.dep_ == "neg" or t.text.lower() in ["no", "never", "not", "n't"]]
    if len(neg_tokens) > 1:
        focus = " ... ".join([t.text for t in neg_tokens])
        report.diagnostics.append(DiagnosticError(
            category="Negative Polarity",
            error_type="Non-Standard Double Negation",
            token_focus=focus,
            suggested_correction="Use single negation with standard negative polarity item (e.g., anything / anybody)",
            linguistic_rule="Standard English syntax enforces single clausal negation or NPI licensing."
        ))
        trace.append("Violation: Multiple negative operators detected within clausal scope.")

    # 6. Uncountable Nouns & Quantifier Concord
    for token in doc:
        word = token.text.lower()
        if word in UNCOUNTABLE_NOUNS and token.tag_ == "NNS":
            report.diagnostics.append(DiagnosticError(
                category="Nominal Semantics / Countability",
                error_type="Uncountable Noun Pluralization",
                token_focus=token.text,
                suggested_correction=UNCOUNTABLE_NOUNS[word],
                linguistic_rule=f"Mass noun '{word}' cannot take plural inflection in standard English."
            ))
            trace.append(f"Violation: Mass noun '{word}' carries plural morpheme.")
        
        # Check determiner/quantifier clash with mass nouns (e.g. many information)
        if word in UNCOUNTABLE_NOUNS:
            for child in token.children:
                if child.dep_ in ["amod", "det"] and child.text.lower() in PLURAL_QUANTIFIERS:
                    report.diagnostics.append(DiagnosticError(
                        category="Nominal Semantics / Countability",
                        error_type="Quantifier-Noun Countability Clash",
                        token_focus=f"{child.text} + {token.text}",
                        suggested_correction=f"much / a lot of {UNCOUNTABLE_NOUNS[word]}",
                        linguistic_rule=f"Countable quantifier '{child.text}' cannot modify mass noun '{token.text}'."
                    ))
                    trace.append("Violation: Countable quantifier bound to non-count noun.")

    # 7. Adjective Ordering Constraints (e.g. red big car -> big red car)
    for token in doc:
        if token.pos_ == "NOUN":
            amods = [child for child in token.children if child.dep_ == "amod"]
            if len(amods) >= 2:
                for i in range(len(amods) - 1):
                    first, second = amods[i].text.lower(), amods[i+1].text.lower()
                    if first in COLOR_WORDS and second in SIZE_WORDS:
                        report.diagnostics.append(DiagnosticError(
                            category="NP Internal Syntax",
                            error_type="Adjective Ordering Constraint Violation",
                            token_focus=f"{first} {second} {token.text}",
                            suggested_correction=f"{second} {first} {token.text}",
                            linguistic_rule="English prenominal modifier hierarchy mandates size adjectives precede color adjectives."
                        ))
                        trace.append(f"Violation: Inverted adjective hierarchy ({first} before {second}).")

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

# Sidebar
with st.sidebar:
    st.markdown("### 🔬 Grammar Engine Scope")
    st.caption("Universal Dependencies L2 Diagnostic Engine v2.5")
    
    with st.expander("Covered Linguistic Phenomona", expanded=True):
        st.markdown("""
        - **SVA:** 3sg Present Concord (`VBZ` vs `VBP`)
        - **Collocation:** Prepositional Complementation
        - **Temporal:** Deictic Adverbial vs Finite Verb
        - **Auxiliaries:** Do-Support Double Marking
        - **Polarity:** Negative Concord Isolation
        - **Countability:** Mass Nouns & Quantifier Clash
        - **Prenominal Syntax:** Adjective Order Constraints
        """)
        
    st.markdown("---")
    st.markdown("#### Test Utterance Presets")
    test_corpora = {
        "Subject-Verb Concord Clash": "He go to the university library every afternoon.",
        "Uncountable Noun Pluralization": "The professor gave us many useful informations.",
        "Adjective Order Constraint": "She bought a red big backpack yesterday.",
        "Erroneous Preposition Selection": "Many students depend of their family for tuition.",
        "Tense-Aspect Clash with Past Adverb": "She writes an academic essay yesterday.",
        "Double Marking Error (Auxiliary)": "The applicant does not has enough credits.",
        "Non-Standard Double Negation": "The student didn't write nothing during the lecture.",
        "Well-Formed Academic Utterance": "The researcher depends on primary data and gathered significant evidence."
    }
    selected_preset = st.selectbox("Select Benchmark Case:", list(test_corpora.keys()))
    preset_query = test_corpora[selected_preset]

# Main Area
st.markdown("""
<div class="main-header">
    <h1 style="margin:0 0 6px 0; color:#ffffff; font-size: 2.1rem; letter-spacing: -0.03em;">L2-SyntaxLens</h1>
    <p style="margin:0; color:#94a3b8; font-size: 0.98rem;">Pedagogical Dependency & Morphosyntactic Error Diagnostic Workbench | Computational Linguistics Lab</p>
</div>
""", unsafe_allow_html=True)

user_utterance = st.text_input("Input L2 Learner Sentence:", value=preset_query)

if user_utterance:
    report, doc, latency = diagnose_l2_syntax(user_utterance)

    col1, col2 = st.columns([3, 1])
    with col1:
        if report.status == "GRAMMATICAL":
            badge_html = '<span class="status-badge badge-valid">✓ VALID SYNTAX (WELL-FORMED)</span>'
        else:
            badge_html = f'<span class="status-badge badge-error">⚠ {report.error_count} SYNTACTIC ANOMALY FLAGGED</span>'
        st.markdown(f"{badge_html} &nbsp;&nbsp; <span style='color:#94a3b8;'>Diagnostic:</span> <code>{report.pedagogical_feedback}</code>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align:right; color:#64748b; font-family:monospace; padding-top:6px;'>Engine Latency: <b style='color:#38bdf8;'>{latency:.2f} ms</b></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    tab_diag, tab_syntax, tab_tokens, tab_schema, tab_trace = st.tabs([
        "Diagnostic Report",
        "Syntactic Dependency Tree", 
        "Token Attributes (Universal Dependencies)",
        "Structured Schema (JSON)", 
        "Rule Traversal Log"
    ])

    with tab_diag:
        st.markdown("#### Morphosyntactic Violations")
        if report.diagnostics:
            for diag in report.diagnostics:
                st.markdown(f"""
                <div class="error-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <h4 style="margin:0; color:#a5b4fc; font-size:1.05rem;">{diag.error_type}</h4>
                        <span style="font-size:0.75rem; background:#2e1065; color:#c084fc; padding:2px 8px; border-radius:4px; font-weight:600;">{diag.category}</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span style="color:#94a3b8; font-size:0.9rem;">Focal Node:</span> <span class="token-chip">{diag.token_focus}</span> 
                        &nbsp;&nbsp;➔&nbsp;&nbsp; 
                        <span style="color:#94a3b8; font-size:0.9rem;">Target Correction:</span> <span class="target-chip">{diag.suggested_correction}</span>
                    </div>
                    <div style="margin-top:10px; font-size:0.87rem; color:#cbd5e1; border-top:1px solid #312e81; padding-top:8px;">
                        <strong>Linguistic Rationale:</strong> <em>{diag.linguistic_rule}</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No syntactic violations identified. Utterance satisfies modeled grammatical constraints.")

    with tab_syntax:
        st.markdown("#### Dependency Parse Graph (Universal Dependencies)")
        svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 115, "color": "#f1f5f9", "bg": "transparent"})
        st.markdown(f'<div class="dep-tree-container">{svg}</div>', unsafe_allow_html=True)

    with tab_tokens:
        st.markdown("#### Token-Level Morphosyntactic Attributes")
        token_records = [{
            "Token": t.text,
            "Lemma": t.lemma_,
            "UPOS": t.pos_,
            "XPOS": t.tag_,
            "Dep Rel": t.dep_,
            "Head Token": t.head.text,
            "Morphology": str(t.morph) or "—"
        } for t in doc]
        st.dataframe(token_records, use_container_width=True, hide_index=True)

    with tab_schema:
        st.markdown("#### Pydantic Diagnostic Serialization")
        st.json(report.model_dump())

    with tab_trace:
        st.markdown("#### Deterministic Rule Traversal Log")
        trace_text = "\n".join([f"[{i+1:02d}] {step}" for i, step in enumerate(report.execution_trace)])
        st.markdown(f'<div class="trace-box"><pre style="margin:0; color:#38bdf8;">{trace_text}</pre></div>', unsafe_allow_html=True)
