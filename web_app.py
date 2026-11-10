import streamlit as st
import spacy
from spacy import displacy
from pydantic import BaseModel, Field
from typing import List, Optional
import time

st.set_page_config(
    page_title="LinguaOrder | Dependency Dialogue Workbench",
    page_icon="🔬",
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

class OrderItem(BaseModel):
    item: str
    quantity: int = 1
    container: Optional[str] = None

class DialogueState(BaseModel):
    raw_utterance: str
    intent: str = "ORDER"
    items: List[OrderItem] = Field(default_factory=list)
    negated_items: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    system_response: str = ""
    status: str = "PENDING"

VALID_FLAVORS = {
    "vanilla": ["vanilla"],
    "chocolate": ["chocolate", "choco"],
    "strawberry": ["strawberry"],
    "mango": ["mango"],
    "pistachio": ["pistachio"]
}
VALID_CONTAINERS = ["cone", "cup", "tub", "bowl"]
NUMBER_MAP = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "single": 1, "double": 2, "triple": 3}

def analyze_utterance(text: str, current_pending_item: Optional[OrderItem] = None):
    start_time = time.perf_counter()
    doc = nlp(text)
    state = DialogueState(raw_utterance=text)
    
    if current_pending_item and not any(k in text.lower() for k in ["cancel", "reset"]):
        for container in VALID_CONTAINERS:
            if container in text.lower():
                current_pending_item.container = container
                state.items.append(current_pending_item)
                state.status = "CONFIRMED"
                state.system_response = f"Order updated: {current_pending_item.quantity}x {current_pending_item.item} in a {container}."
                return state, doc, (time.perf_counter() - start_time) * 1000

    for token in doc:
        if token.lemma_ in ["cancel", "stop", "nevermind"]:
            state.intent = "CANCEL"
            state.status = "TERMINATED"
            state.system_response = "Dialogue context cleared."
            return state, doc, (time.perf_counter() - start_time) * 1000

    for token in doc:
        matched_flavor = None
        for canonical, syns in VALID_FLAVORS.items():
            if token.lemma_.lower() in syns or token.text.lower() in syns:
                matched_flavor = canonical
                break
                
        if matched_flavor:
            head = token.head
            is_negated = any(
                child.dep_ == "neg" or child.text.lower() in ["no", "without", "don't", "dont"]
                for child in list(token.children) + list(head.children)
            )
            if is_negated:
                state.negated_items.append(matched_flavor)
                continue

            quantity = 1
            for child in list(token.children) + list(head.children):
                if child.dep_ in ["nummod", "quantmod", "det", "amod"]:
                    if child.text.isdigit():
                        quantity = int(child.text)
                        break
                    elif child.text.lower() in NUMBER_MAP:
                        quantity = NUMBER_MAP[child.text.lower()]
                        break

            detected_container = next((c for c in VALID_CONTAINERS if c in text.lower()), None)

            state.items.append(OrderItem(
                item=matched_flavor,
                quantity=quantity,
                container=detected_container
            ))

    if not state.items and not state.negated_items:
        state.status = "NO_MATCH"
        state.system_response = "No recognized entities in utterance."
    else:
        for it in state.items:
            if not it.container:
                state.missing_slots.append(f"container_for_{it.item}")
                
        if state.missing_slots:
            state.status = "PENDING_INPUT"
            unspecified = [it.item for it in state.items if not it.container]
            state.system_response = f"Please specify container (cup or cone) for: {', '.join(unspecified)}."
        else:
            state.status = "CONFIRMED"
            summary = [f"{it.quantity}x {it.item} ({it.container})" for it in state.items]
            state.system_response = f"Validated parameters: {', '.join(summary)}."

    return state, doc, (time.perf_counter() - start_time) * 1000

with st.sidebar:
    st.markdown("### Domain Specifications")
    st.caption("Grammar-guided slot-filling engine")
    
    st.markdown("**Controlled Lexicon**")
    st.markdown("`Flavors:` vanilla, chocolate, strawberry, mango, pistachio")
    st.markdown("`Containers:` cup, cone, tub, bowl")
    
    st.markdown("---")
    st.markdown("**Corpus Test Cases**")
    presets = {
        "Default Test": "I want two strawberry please",
        "Compound & Quantified": "Two vanilla scoops in a cup and one chocolate in a cone",
        "Negation Test": "I want chocolate in a cup but no strawberry",
        "Termination Test": "Cancel order please"
    }
    selected_name = st.selectbox("Select Test Utterance:", list(presets.keys()))
    preset_query = presets[selected_name]

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

    tab_syntax, tab_state, tab_tokens = st.tabs([
        "Syntactic Dependency Tree", 
        "State Schema (JSON)", 
        "Token-Level Linguistic Features"
    ])

    with tab_syntax:
        st.markdown("##### Dependency Parse Graph (Universal Dependencies)")
        svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 110, "color": "#e2e8f0", "bg": "transparent"})
        st.markdown(f'<div class="dep-tree-container">{svg}</div>', unsafe_allow_html=True)

    with tab_state:
        st.markdown("##### Dialogue State Representation (Pydantic Output)")
        st.json(state.model_dump())

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
