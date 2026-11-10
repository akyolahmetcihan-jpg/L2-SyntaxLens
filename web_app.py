import time
from typing import List, Optional
from pydantic import BaseModel, Field
import spacy
from spacy import displacy
import streamlit as st

st.set_page_config(
    page_title="L2-SyntaxLens | Diagnostic Workbench",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .dep-tree-container {
        background-color: #1a1c23;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 20px;
        overflow-x: auto;
    }
    .dep-tree-container svg {
        color: #e2e8f0 !important;
        font-family: monospace;
    }
    .dep-tree-container text {
        fill: #e2e8f0 !important;
    }
    .dep-tree-container path, .dep-tree-container line {
        stroke: #64748b !important;
    }
    .badge-error {
        background-color: #4c0519;
        color: #fda4af;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #9f1239;
    }
    .badge-valid {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #047857;
    }
    .trace-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.82rem;
        color: #38bdf8;
    }
    .error-card {
        background-color: #1e1b4b;
        border-left: 4px solid #6366f1;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
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
    "graduate": "from"
}

PAST_TIME_ADVERBIALS = [
    "yesterday", "ago", "last", "formerly", "previously",
    "in 2020", "in 2021", "in 2022"
]

def diagnose_l2_syntax(text: str):
    start_time = time.perf_counter()
    doc = nlp(text)
    report = SyntaxDiagnosticReport(raw_utterance=text, status="GRAMMATICAL")
    trace = report.execution_trace

    trace.append(f"Tokenized utterance into {len(doc)} tokens.")

    # 1. Özne-Yüklem Uyuşmazlığı (Subject-Verb Agreement)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ in ["VERB", "AUX"]:
            verb = token.head
            trace.append(f"Checking nsubj('{token.text}') -> head('{verb.text}')")
            is_3sg = (
                token.lower_ in ["he", "she", "it"] or 
                (token.tag_ in ["NN", "NNP"] and token.text.lower() not in ["i", "you"])
            )
            if is_3sg and verb.tag_ == "VBP":
                report.diagnostics.append(DiagnosticError(
                    category="Morphosyntactic Agreement",
                    error_type="Subject-Verb Concord Error",
                    token_focus=token.text + " + " + verb.text,
                    suggested_correction=token.text + " " + verb.lemma_ + "s",
                    linguistic_rule="Third-person singular subjects select for 3sg present inflection (-s, XPOS: VBZ)."
                ))
                trace.append("Violation: SVA clash between 3sg subject and uninflected verb.")

    # 2. Yanlış Edat Seçimi (Prepositional Collocation)
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
                            token_focus=token.text + " + " + child.text,
                            suggested_correction=token.text + " " + target_prep,
                            linguistic_rule=f"Predicate '{lemma}' selects preposition '{target_prep}' via complementation."
                        ))
                        trace.append(f"Violation: '{lemma}' paired with '{child.text}' instead of '{target_prep}'.")

    # 3. Zaman Zarfı & Zaman Uyuşmazlığı (Tense-Aspect Clash)
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
                    token_focus=token.text + " ... " + past_anchor,
                    suggested_correction="Inflect '" + token.text + "' into past tense (VBD)",
                    linguistic_rule=f"Past temporal deictic anchor '{past_anchor}' requires simple past morphology."
                ))
                trace.append(f"Violation: Present verb '{token.text}' clashed with past anchor '{past_anchor}'.")

    # 4. Çifte Çekimleme (Double Marking / Do-Support)
    for token in doc:
        if token.dep_ == "aux" and token.lemma_ == "do":
            main_verb = token.head
            if main_verb.pos_ == "VERB" and main_verb.tag_ in ["VBZ", "VBD"]:
                report.diagnostics.append(DiagnosticError(
                    category="Auxiliary Complementation",
                    error_type="Double Marking / Auxiliary Redundancy",
                    token_focus=token.text + " + " + main_verb.text,
                    suggested_correction=token.text + " " + main_verb.lemma_,
                    linguistic_rule="Do-support absorbs inflection; lexical verb must be bare infinitive (VB)."
                ))
                trace.append("Violation: Double marking detected on both auxiliary and main verb.")

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
    st.markdown("### Grammar Rules Scope")
    st.caption("Universal Dependencies L2 Diagnostic Engine")
    st.markdown("- `SVA:` Subject-Verb Concord (`VBZ` vs `VBP`)")
    st.markdown("- `Collocations:` Prepositional Verbs (`depend on`, `rely on`, etc.)")
    st.markdown("- `Temporal:` Past Adverbial vs Present Verb Tense Clash")
    st.markdown("- `Auxiliaries:` Do-Support & Bare Infinitive Complementation")
    st.markdown("---")
    st.markdown("**Test Corpora Presets**")
    test_corpora = {
        "Subject-Verb Agreement Clash": "He go to the university library every afternoon.",
        "Erroneous Preposition Selection": "Many students depend of their family for tuition.",
        "Tense-Aspect Clash with Past Adverb": "She writes an academic essay yesterday.",
        "Double Marking Error (Auxiliary)": "The applicant does not has enough credits.",
        "Well-Formed Control Sentence": "The student depends on primary sources and wrote an essay yesterday."
    }
    selected_preset = st.selectbox("Select Test Utterance:", list(test_corpora.keys()))
    preset_query = test_corpora[selected_preset]

# Main Area
st.title("L2-SyntaxLens")
st.caption("Pedagogical Dependency & Morphosyntactic Error Diagnostic Workbench | Computational Linguistics Laboratory")

user_utterance = st.text_input("Input L2 Learner Sentence:", value=preset_query)

if user_utterance:
    report, doc, latency = diagnose_l2_syntax(user_utterance)

    if report.status == "GRAMMATICAL":
        badge_html = '<span class="badge-valid">VALID SYNTAX (WELL-FORMED)</span>'
    else:
        badge_html = f'<span class="badge-error">{report.error_count} SYNTACTIC ANOMALY DETECTED</span>'

    st.markdown(f"""
    **Status:** {badge_html} &nbsp;|&nbsp; 
    **Diagnostic:** `{report.pedagogical_feedback}` &nbsp;|&nbsp; 
    **Latency:** `{latency:.2f} ms`
    """, unsafe_allow_html=True)

    st.markdown("---")

    tab_diag, tab_syntax, tab_schema, tab_trace, tab_tokens = st.tabs([
        "Diagnostic Report",
        "Syntactic Dependency Tree", 
        "Report Schema (JSON)", 
        "Rule Traversal Log",
        "Token Attributes (Universal Dependencies)"
    ])

    with tab_diag:
        st.markdown("##### Pedagogical & Morphosyntactic Diagnostics")
        if report.diagnostics:
            for diag in report.diagnostics:
                st.markdown(f"""
                <div class="error-card">
                    <h4 style="margin:0 0 6px 0; color:#818cf8;">⚠️ {diag.error_type} <span style="font-size:0.8rem; color:#94a3b8;">({diag.category})</span></h4>
                    <p style="margin:2px 0;"><strong>Focal Node:</strong> <code>{diag.token_focus}</code> &nbsp;➔&nbsp; <strong>Suggested Target:</strong> <code>{diag.suggested_correction}</code></p>
                    <p style="margin:6px 0 0 0; font-size:0.88rem; color:#cbd5e1;"><em><strong>Linguistic Rationale:</strong> {diag.linguistic_rule}</em></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No morphosyntactic violations identified. The syntactic parse tree demonstrates grammatical concordance.")

    with tab_syntax:
        st.markdown("##### Dependency Parse Graph (Universal Dependencies)")
        svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 110, "color": "#e2e8f0", "bg": "transparent"})
        st.markdown(f'<div class="dep-tree-container">{svg}</div>', unsafe_allow_html=True)

    with tab_schema:
        st.markdown("##### Pydantic Structured Diagnostic Schema")
        st.json(report.model_dump())

    with tab_trace:
        st.markdown("##### Step-by-Step Rule Traversal Log")
        trace_text = "\n".join([f"[{i+1:02d}] {step}" for i, step in enumerate(report.execution_trace)])
        st.markdown(f'<div class="trace-box"><pre style="margin:0; color:#38bdf8;">{trace_text}</pre></div>', unsafe_allow_html=True)

    with tab_tokens:
        st.markdown("##### Token-Level Morphosyntactic Attributes")
        token_records = [{
            "Token": t.text,
            "Lemma": t.lemma_,
            "UPOS": t.pos_,
            "XPOS (Penn Treebank)": t.tag_,
            "Dep Rel": t.dep_,
            "Head Token": t.head.text,
            "Morphology": str(t.morph) or "—"
        } for t in doc]
        st.dataframe(token_records, use_container_width=True, hide_index=True)
