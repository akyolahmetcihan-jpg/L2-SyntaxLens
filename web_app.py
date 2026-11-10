import streamlit as st
import spacy
from spacy import displacy
from pydantic import BaseModel, Field
from typing import List, Optional
import time

st.set_page_config(
    page_title="L2-SyntaxLens | Pedagogical Error Diagnostic Workbench",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
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

PAST_TIME_ADVERBIALS = ["yesterday", "ago", "last", "formerly", "previously", "in 2020", "in 2021", "in 2022"]

def diagnose_l2_syntax(text: str):
    start_time = time.perf_counter()
    doc = nlp(text)
    report = SyntaxDiagnosticReport(raw_utterance=text, status="GRAMMATICAL")
    trace = report.execution_trace

    trace.append(f"Segmented and tokenized into {len(doc)} tokens.")

    # 1. Özne-Yüklem Uyuşmazlığı (Subject-Verb Agreement)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ in ["VERB", "AUX"]:
            verb = token.head
            trace.append(f"Analyzing dependency relation: nsubj('{token.text}') -> head('{verb.text}')")
            is_third_person_singular = token.lower_ in ["he", "she", "it"] or (token.tag_ in ["NN", "NNP"] and token.text.lower() not in ["i", "you"])
            
            if is_third_person_singular and verb.tag_ == "VBP":
                report.diagnostics.append(DiagnosticError(
                    category="Morphosyntactic Agreement",
                    error_type="Subject-Verb Concord Error",
                    token_focus=f"{token.text} + {verb.text}",
                    suggested_correction=f"{token.text} {verb.lemma_}s",
                    linguistic_rule="Third-person singular subjects (he/she/it/NN) require 3rd-person singular present inflection (-s, XPOS: VBZ)."
                ))
                trace.append(f"Violation flagged: SVA clash between '{token.text}' and base present verb '{verb.text}'.")

    # 2. Yanlış Edat Seçimi (Preposition Collocation Errors)
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"]:
            lemma = token.lemma_.lower()
            if lemma in VERB_PREP_COLLOCATIONS:
                expected_prep = VERB_PREP_COLLOCATIONS[lemma]
                for child in token.children:
                    if child.dep_ in ["prep", "adp"] and child.text.lower() != expected_prep:
                        report.diagnostics.append(DiagnosticError(
                            category="Lexicosyntactic Collocation",
                            error_type="Erroneous Prepositional Selection",
                            token_focus=f"{token.text} + {child.
