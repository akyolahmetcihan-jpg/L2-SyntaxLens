import streamlit as st
import spacy
from spacy import displacy
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time

st.set_page_config(
    page_title="LinguaOrder | Dependency Dialogue Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema & Düzen
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
    .badge-confirmed {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #047857;
    }
    .badge-pending {
        background-color: #451a03;
        color: #fcd34d;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #78350f;
    }
    .badge-neutral {
        background-color: #1e293b;
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .trace-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.82rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# --- GELİŞMİŞ PYDANTIC ŞEMALARI ---
class OrderItem(BaseModel):
    item: str
    canonical_match: str
    quantity: int = 1
    container: Optional[str] = None
    modifiers: List[str] = Field(default_factory=list)
    confidence: float = 1.0

class DialogueState(BaseModel):
    raw_utterance: str
    intent: str = "ORDER"
    items: List[OrderItem] = Field(default_factory=list)
    negated_items: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)
    system_response: str = ""
    status: str = "PENDING"

# --- GELİŞMİŞ SÖZLÜK VE SYNONYM TABLOSU ---
LEXICON_FLAVORS = {
    "vanilla": ["vanilla", "french-vanilla"],
    "chocolate": ["chocolate", "choco", "cocoa", "fudge"],
    "strawberry": ["strawberry", "berries", "berry"],
    "mango": ["mango", "tropical"],
    "pistachio": ["pistachio", "nut"]
}

LEXICON_CONTAINERS = {
    "cone": ["cone", "waffle", "sugar-cone", "cornet"],
    "cup": ["cup", "tub", "bowl", "dish"]
}

LEXICON_TOPPINGS = ["drizzle", "sprinkles", "sauce", "syrup", "nuts", "flakes", "cream"]

NUMBER_MAP = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "single": 1, "double": 2, "triple": 3
}

def analyze_utterance(text: str, current_pending_item: Optional[OrderItem] = None):
    start_time = time.perf_counter()
    doc = nlp(text)
    state = DialogueState(raw_utterance=text)
    trace = state.execution_trace

    trace.append(f"Tokenized utterance into {len(doc)} tokens.")

    # 1. Eksik Parametre Modu (Multi-turn Slot Filling)
    if current_pending_item and not any(k in text.lower() for k in ["cancel", "stop", "reset"]):
        trace.append(f"Active slot recovery: Resolving container for '{current_pending_item.item}'")
        for canonical, syns in LEXICON_CONTAINERS.items():
            if any(syn in text.lower() for syn in syns):
                current_pending_item.container = canonical
                state.items.append(current_pending_item)
                state.status = "CONFIRMED"
                state.system_response = f"Order updated: {current_pending_item.quantity}x {current_pending_item.canonical_match} in a {canonical}."
                trace.append(f"Successfully bound container '{canonical}' via multi-turn context.")
                return state, doc, (time.perf_counter() - start_time) * 1000

    # 2. İptal ve Durdurma Niyeti (Intent Detection via Lemma)
    for token in doc:
        if token.lemma_ in ["cancel", "stop", "nevermind", "abort"]:
            state.intent = "CANCEL"
            state.status = "TERMINATED"
            state.system_response = "Dialogue context cleared and order aborted."
            trace.append(f"Intent classified as 'CANCEL' via root/modifier token '{token.text}' (lemma: '{token.lemma_}').")
            return state, doc, (time.perf_counter() - start_time) * 1000

    # 3. Sentaktik Ağaç Taraması (Dependency Tree Traversal)
    for token in doc:
        # A. Ürün Tespiti
        matched_canonical = None
        for canonical, syns in LEXICON_FLAVORS.items():
            if token.lemma_.lower() in syns or token.text.lower() in syns:
                matched_canonical = canonical
                break

        if matched_canonical:
            trace.append(f"Found lexical candidate '{token.text}' mapped to canonical '{matched_canonical}'.")
            head = token.head

            # B. Sentaktik Olumsuzluk Analizi ('neg' bağıntısı ve yerel subtree)
            subtree_tokens = list(token.subtree)
            is_negated = any(
                child.dep_ == "neg" or child.text.lower() in ["no", "without", "don't", "dont"]
                for child in list(token.children) + list(head.children) + subtree_tokens
            )

            if is_negated:
                state.negated_items.append(matched_canonical)
                trace.append(f"Negation scope triggered: Node '{token.text}' excluded via syntactic negation dependency.")
                continue

            # C. Nicelik / Adet Tespiti (nummod veya amod bağları)
            quantity = 1
            num_source = "default (1)"
            for child in list(token.children) + list(head.children):
                if child.dep_ in ["nummod", "quantmod", "det", "amod"]:
                    if child.text.isdigit():
                        quantity = int(child.text)
                        num_source = f"nummod child '{child.text}'"
                        break
                    elif child.text.lower() in NUMBER_MAP:
                        quantity = NUMBER_MAP[child.text.lower()]
                        num_source = f"lexical quantifier '{child.text}'"
                        break

            trace.append(f"Bound quantity {quantity} to '{matched_canonical}' via {num_source}.")

            # D. Kap Türü Eşleştirme (Lexical & Prepositional Attachment)
            detected_container = None
            for canonical_c, syns in LEXICON_CONTAINERS.items():
                if any(syn in text.lower() for syn in syns):
                    detected_container = canonical_c
                    break

            # E. Niteleyiciler ve Topping Tespiti (Prepositional Objects: 'prep' -> 'pobj')
            detected_modifiers = []
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "with":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            detected_modifiers.append(pobj.text)
            
            # Doğrudan eşleşen toppingler
            for top in LEXICON_TOPPINGS:
                if top in text.lower() and top not in detected_modifiers:
                    detected_modifiers.append(top)

            item = OrderItem(
                item=token.text,
                canonical_match=matched_canonical,
                quantity=quantity,
                container=detected_container,
                modifiers=detected_modifiers,
                confidence=0.98 if matched_canonical == token.text.lower() else 0.90
            )
            state.items.append(item)

    # 4. Durum Yönetimi & Eksik Parametre Tespiti
    if not state.items and not state.negated_items:
        state.status = "NO_MATCH"
        state.system_response = "No recognized flavors in utterance."
        trace.append("Parser completed with zero valid entity extractions.")
    else:
        for it in state.items:
            if not it.container:
                state.missing_slots.append(f"container_for_{it.canonical_match}")

        if state.missing_slots:
            state.status = "PENDING_INPUT"
            unspecified = [it.canonical_match for it in state.items if not it.container]
            state.system_response = f"Please specify container (cup or cone) for: {', '.join(unspecified)}."
            trace.append(f"Slot filler triggered: Missing container specification for {unspecified}.")
        else:
            state.status = "CONFIRMED"
            summary = [f"{it.quantity}x {it.canonical_match} ({it.container})" + (f" with {', '.join(it.modifiers)}" if it.modifiers else "") for it in state.items]
            state.system_response = f"Validated order: {', '.join(summary)}."
            trace.append("All required parameters resolved. Emitting validated transaction schema.")

    latency = (time.perf_counter() - start_time) * 1000
    return state, doc, latency

# --- SIDEBAR (KONTROL PANELİ) ---
with st.sidebar:
    st.markdown("### Domain Specifications")
    st.caption("Universal Dependencies Slot-Filling Core")

    st.markdown("**Controlled Lexicon & Synonyms**")
    st.markdown("- `Flavors:` vanilla, chocolate (fudge/cocoa), strawberry, mango, pistachio")
    st.markdown("- `Containers:` cup (bowl/tub), cone (waffle/cornet)")
    st.markdown("- `Modifiers:` drizzle, sprinkles, sauce, nuts, syrup")

    st.markdown("---")
    st.markdown("**Corpus Test Cases**")
    presets = {
        "Compound Order with Modifiers": "Two vanilla scoops in a cup with chocolate drizzle and one strawberry in a cone",
        "Synonym & Container Variant": "I want a single cocoa in a waffle please",
        "Negation Subtree Test": "Give me two mangoes in a bowl but without strawberry",
        "Ambiguity & Slot Recovery": "Can I get three scoops of pistachio please?",
        "Dialogue Abortion": "Actually stop, cancel my order please"
    }
    selected_name = st.selectbox("Select Test Utterance:", list(presets.keys()))
    preset_query = presets[selected_name]

# --- ANA EKRAN ---
st.title("LinguaOrder Workbench")
st.caption("Syntactic Dependency Dialogue System | Computational Linguistics Laboratory")

if "pending_item" not in st.session_state:
    st.session_state.pending_item = None

user_utterance = st.text_input("Input Utterance:", value=preset_query)

if user_utterance:
    state, doc, latency = analyze_utterance(user_utterance, st.session_state.pending_item)

    if state.status == "PENDING_INPUT":
        st.session_state.pending_item = [it for it in state.items if not it.container][0]
    elif state.status == "CONFIRMED":
        st.session_state.pending_item = None

    if state.status == "CONFIRMED":
        badge_html = '<span class="badge-confirmed">CONFIRMED</span>'
    elif state.status == "PENDING_INPUT":
        badge_html = '<span class="badge-pending">SLOT FILLING REQUIRED</span>'
    else:
        badge_html = f'<span class="badge-neutral">{state.status}</span>'

    st.markdown(f"""
    **Status:** {badge_html} &nbsp;|&nbsp; 
    **Response:** `{state.system_response}` &nbsp;|&nbsp; 
    **Latency:** `{latency:.2f} ms`
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sekmeler
    tab_syntax, tab_state, tab_trace, tab_tokens = st.tabs([
        "Syntactic Dependency Tree", 
        "State Schema (JSON)", 
        "Parser Execution Trace",
        "Token-Level Linguistic Features"
    ])

    with tab_syntax:
        st.markdown("##### Dependency Parse Graph (Universal Dependencies)")
        svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 110, "color": "#e2e8f0", "bg": "transparent"})
        st.markdown(f'<div class="dep-tree-container">{svg}</div>', unsafe_allow_html=True)

    with tab_state:
        st.markdown("##### Dialogue State Representation (Pydantic Output)")
        st.json(state.model_dump())

    with tab_trace:
        st.markdown("##### Step-by-Step Rule Traversal Log")
        trace_text = "\n".join([f"[{i+1:02d}] {step}" for i, step in enumerate(state.execution_trace)])
        st.markdown(f'<div class="trace-box"><pre style="margin:0; color:#38bdf8;">{trace_text}</pre></div>', unsafe_allow_html=True)

    with tab_tokens:
        st.markdown("##### Morphosyntactic Attributes")
        token_records = [{
            "Token": t.text,
            "Lemma": t.lemma_,
            "UPOS": t.pos_,
            "XPOS": t.tag_,
            "Dep Rel": t.dep_,
            "Head": t.head.text,
            "Children": ", ".join([c.text for c in t.children]) or "—"
        } for t in doc]
        st.dataframe(token_records, use_container_width=True, hide_index=True)
