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
    "graduate": "from",
    "agree": "with",
    "consist": "of"
}

PAST_TIME_ADVERBIALS = [
    "yesterday", "ago", "last", "formerly", "previously",
    "in 2020", "in 2021", "in 2022", "in 2023", "in 2024"
]

PLURAL_DETERMINERS = ["these", "those", "many", "several", "few"]
SINGULAR_DETERMINERS = ["this", "that", "each", "every", "a", "an"]

def diagnose_l2_syntax(text: str):
    start_time = time.perf_counter()
    doc = nlp(text)
    report = SyntaxDiagnosticReport(raw_utterance=text, status="GRAMMATICAL")
    trace = report.execution_trace

    trace.append(f"Segmented utterance into {len(doc)} tokens.")

    # 1. Subject-Verb Agreement (SVA)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ in ["VERB", "AUX"]:
            verb = token.head
            trace.append(f"Checking agreement: nsubj('{token.text}') -> head('{verb.text}')")
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
                            token_focus=token.text + " + " + child.text,
                            suggested_correction=token.text + " " + target_prep,
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
                    token_focus=token.text + " ... " + past_anchor,
                    suggested_correction="Inflect '" + token.text + "' into past tense (VBD)",
                    linguistic_rule=f"Past deictic anchor '{past_anchor}' requires simple past morphology."
                ))
                trace.append(f"Violation: Present verb '{token.text}' clashed with past anchor '{past_anchor}'.")

    # 4. Double Marking / Auxiliary Complementation
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

    # 5. Determiner-Noun Number Concordance (NP Agreement)
    for token in doc:
        if token.dep_ == "det":
            det_text = token.text.lower()
            head_noun = token.head
            if head_noun.pos_ == "NOUN":
                if det_text in PLURAL_DETERMINERS and head_noun.tag_ == "NN":
                    report.diagnostics.append(DiagnosticError(
                        category="NP Morphosyntax",
                        error_type="Determiner-Noun Number Mismatch",
                        token_focus=token.text + " + " + head_noun.text,
                        suggested_correction=token.text + " " + head_noun.lemma_ + "s",
                        linguistic_rule=f"Plural determiner '{det_text}' requires plural head noun morphology (NNS)."
                    ))
                    trace.append("Violation: Plural determiner attached to singular count noun.")
                elif det_text in ["these", "those"] and head_noun.tag_ == "NN":
                    report.diagnostics.append(DiagnosticError(
                        category="NP Morphosyntax",
                        error_type="Demonstrative Concord Clash",
                        token_focus=token.text + " + " + head_noun.text,
                        suggested_correction="this/that " + head_noun.text,
                        linguistic_rule="Proximal/distal plural demonstratives require plural head complement."
                    ))

    # 6. Non-Standard Double Negation
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
        trace.append("Violation: Multiple negative operators detected within local clausal scope.")

    # 7. Modal Auxiliary Complementation (can to go / must goes)
    for token in doc:
        if token.pos_ == "AUX" and token.tag_ == "MD":
            main_verb = token.head
            for child in token.children:
                if child.text.lower() == "to":
                    report.diagnostics.append(DiagnosticError(
                        category="Modal Syntax",
                        error_type="Infinitival Marker with Modal",
                        token_focus=token.text + " to " + main_verb.text,
                        suggested_correction=token.text + " " + main_verb.text,
                        linguistic_rule="Modal auxiliaries select bare infinitival complements without 'to'."
                    ))
            if main_verb.tag_ in ["VBZ", "VBD", "VBG"]:
                report.diagnostics.append(DiagnosticError(
                    category="Modal Syntax",
                    error_type="Inflected Modal Complement",
                    token_focus=token.text + " + " + main_verb.text,
                    suggested_correction=token.text + " " + main_verb.lemma_,
                    linguistic_rule="Modal auxiliaries strictly govern bare infinitive predicates (VB)."
                ))

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
    st.caption("Universal Dependencies L2 Diagnostic Engine v2.0")
    st.markdown("- `SVA:` 3sg Concord (`VBZ` vs `VBP`)")
    st.markdown("- `Collocations:` Prepositional Selection Frame")
    st.markdown("- `Temporal:` Past Adverbial vs Present Verb Clash")
    st.markdown("- `Auxiliaries:` Do-Support & Bare Infinitive")
    st.markdown("- `NP Syntax:` Determiner-Head Number Agreement")
    st.markdown("- `Polarity:` Double Negation Elimination")
    st.markdown("- `Modals:` Bare Infinitival Complementation")
    st.markdown("---")
    st.markdown("**Test Corpora Presets**")
    test_corpora = {
        "Subject-Verb Agreement Clash": "He go to the university library every afternoon.",
        "Erroneous Preposition Selection": "Many students depend of their family for tuition.",
        "Tense-Aspect Clash with Past Adverb": "She writes an academic essay yesterday.",
        "Double Marking Error (Auxiliary)": "The applicant does not has enough credits.",
        "Determiner-Noun Number Clash": "These book contains fundamental computational theories.",
        "Non-Standard Double Negation": "The student didn't write nothing during the lecture.",
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
