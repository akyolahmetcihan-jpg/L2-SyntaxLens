import streamlit as st
import spacy
from spacy import displacy
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time

# --- SAYFA YAPILANDIRMASI & TEMA ---
st.set_page_config(
    page_title="LinguaOrder AI | Computational Linguistics Workbench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel Arayüz Stili (Modern UI & Metrik Kartları)
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
    .displacy-container {
        overflow-x: auto;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Dil Modelini Önbelleğe Alarak Yükleme
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# --- VERİ YAPILANDIRMA MODELLERİ (PYDANTIC) ---
class OrderItem(BaseModel):
    item: str
    quantity: int = 1
    container: Optional[str] = None
    modifiers: List[str] = Field(default_factory=list)

class DialogueState(BaseModel):
    raw_utterance: str
    intent: str = "ORDER"
    items: List[OrderItem] = Field(default_factory=list)
    negated_items: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    system_response: str = ""
    status: str = "PENDING"

# --- KURAL TABANLI SENTAKTİK DİLBİLİM MOTORU ---
VALID_FLAVORS = {
    "vanilla": ["vanilla"],
    "chocolate": ["chocolate", "choco"],
    "strawberry": ["strawberry", "berry"],
    "mango": ["mango"],
    "pistachio": ["pistachio"]
}
VALID_CONTAINERS = ["cone", "cup", "tub", "bowl"]
NUMBER_MAP = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, 
    "four": 4, "five": 5, "single": 1, "double": 2, "triple": 3
}

def analyze_utterance(text: str, current_pending_item: Optional[OrderItem] = None) -> (DialogueState, any, float):
    start_time = time.perf_counter()
    doc = nlp(text)
    state = DialogueState(raw_utterance=text)
    
    # Çok adımlı diyalog (Eksik parametre tamamlama modu)
    if current_pending_item and not any(k in text.lower() for k in ["cancel", "reset", "stop"]):
        for container in VALID_CONTAINERS:
            if container in text.lower():
                current_pending_item.container = container
                state.items.append(current_pending_item)
                state.status = "CONFIRMED"
                state.system_response = f"Perfect! Added {current_pending_item.quantity} {current_pending_item.item} in a {container}."
                latency = (time.perf_counter() - start_time) * 1000
                return state, doc, latency

    # 1. Kök Fiil ve Niyet Analizi
    for token in doc:
        if token.lemma_ in ["cancel", "nevermind", "stop"]:
            state.intent = "CANCEL_ORDER"
            state.status = "TERMINATED"
            state.system_response = "Order processing cancelled."
            latency = (time.perf_counter() - start_time) * 1000
            return state, doc, latency

    # 2. Sentaktik Bağımlılık & Slot-Filling
    for token in doc:
        # Menü öğesi eşleştirme
        matched_flavor = None
        for canonical, syns in VALID_FLAVORS.items():
            if token.lemma_.lower() in syns or token.text.lower() in syns:
                matched_flavor = canonical
                break
                
        if matched_flavor:
            # Olumsuzluk Kontrolü (Syntactic Negation: 'neg' dependency)
            is_negated = False
            head = token.head
            # Doğrudan çocuğunda veya bağlı olduğu üst fiilde 'no/not' var mı?
            for child in list(token.children) + list(head.children):
                if child.dep_ == "neg" or child.text.lower() in ["no", "without", "don't", "dont"]:
                    is_negated = True
                    break
                    
            if is_negated:
                state.negated_items.append(matched_flavor)
                continue

            # Miktar / Sayı Belirteci Tespiti (nummod veya amod)
            quantity = 1
            search_scope = list(token.children) + list(head.children)
            for child in search_scope:
                if child.dep_ in ["nummod", "quantmod", "det", "amod"]:
                    if child.text.isdigit():
                        quantity = int(child.text)
                        break
                    elif child.text.lower() in NUMBER_MAP:
                        quantity = NUMBER_MAP[child.text.lower()]
                        break

            # Kap / Format Tespiti (Prepositional Objects: 'prep' -> 'pobj')
            detected_container = None
            for container in VALID_CONTAINERS:
                if container in text.lower():
                    detected_container = container
                    break

            order_item = OrderItem(
                item=matched_flavor,
                quantity=quantity,
                container=detected_container
            )
            state.items.append(order_item)

    # 3. Durum ve Eksik Slot Yönetimi
    if not state.items and not state.negated_items:
        state.status = "NO_INTENT_DETECTED"
        state.system_response = "No menu items recognized. Please specify an available flavor (e.g., vanilla, chocolate, strawberry)."
    else:
        for it in state.items:
            if not it.container:
                state.missing_slots.append(f"container_for_{it.item}")
                
        if state.missing_slots:
            state.status = "SLOT_FILLING_REQUIRED"
            unspecified_items = [it.item for it in state.items if not it.container]
            state.system_response = f"Would you like your {', '.join(unspecified_items)} in a cup or a cone?"
        else:
            state.status = "CONFIRMED"
            summary = [f"{it.quantity}x {it.item} ({it.container})" for it in state.items]
            state.system_response = f"Order successfully validated: {', '.join(summary)}."

    latency = (time.perf_counter() - start_time) * 1000
    return state, doc, latency


# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Engine Architecture")
    st.markdown("**Rule-Grounded Syntactic Dialogue Agent**")
    st.caption("Computational Linguistics Core v2.1")
    
    st.markdown("---")
    st.subheader("📚 Lexical Domain")
    st.markdown("- **Flavors:** `vanilla`, `chocolate`, `strawberry`, `mango`, `pistachio`")
    st.markdown("- **Containers:** `cup`, `cone`, `tub`, `bowl`")
    st.markdown("- **Numerical Quantifiers:** Words (`two`, `double`) & Digits (`2`)")
    
    st.markdown("---")
    st.subheader("💡 Test Örnekleri")
    test_presets = [
        "I'd love two scoops of pistachio in a cup and one chocolate in a cone.",
        "Can I get three vanilla scoops please?",
        "I want a double chocolate in a bowl but no strawberry.",
        "Cancel the order, please."
    ]
    selected_preset = st.selectbox("Hızlı Senaryo Seç:", ["-- Boş / Kendi Cümleni Yaz --"] + test_presets)

# --- ANA EKRAN DÜZENİ ---
st.title("🍦 LinguaOrder AI")
st.markdown("#### **Computational Linguistics & Dependency-Grounded Dialogue Workbench**")
st.markdown("An explicit rule-grounded NLP agent leveraging syntactic dependency trees (`spaCy`), strict schema validation (`Pydantic`), and finite-state slot-filling.")

st.divider()

# Session State ile Çok Adımlı Konuşma Takibi
if "pending_item" not in st.session_state:
    st.session_state.pending_item = None

initial_text = selected_preset if selected_preset != "-- Boş / Kendi Cümleni Yaz --" else "I want two strawberry please"
user_utterance = st.text_input("Girdi Cümlesi (Customer Natural Language Utterance):", value=initial_text)

col_act1, col_act2 = st.columns([1, 5])
with col_act1:
    submit_btn = st.button("🚀 Parse & Execute", type="primary", use_container_width=True)
with col_act2:
    if st.session_state.pending_item:
        st.info(f"⏳ **Açık Diyalog Durumu:** `{st.session_state.pending_item.item}` için kap/külah seçimi bekleniyor...")

if user_utterance:
    state, doc, latency = analyze_utterance(user_utterance, st.session_state.pending_item)
    
    # State güncelleme
    if state.status == "SLOT_FILLING_REQUIRED":
        st.session_state.pending_item = [it for it in state.items if not it.container][0]
    elif state.status == "CONFIRMED":
        st.session_state.pending_item = None

    # 1. Mühendislik Metrikleri Şeridi
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Çözümleme Süresi (Latency)", f"{latency:.2f} ms")
    with m2:
        st.metric("Durum (Dialogue State)", state.status)
    with m3:
        st.metric("Geçerli Ürün Adedi", len(state.items))
    with m4:
        st.metric("Dışlanan (Negated) Öğe", len(state.negated_items))

    st.markdown("---")

    # 2. İki Sütunlu Detaylı İnceleme
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("💬 Ajan Yanıtı & Durum")
        if state.status == "CONFIRMED":
            st.success(f"**Agent:** {state.system_response}")
        elif state.status == "SLOT_FILLING_REQUIRED":
            st.warning(f"**Agent:** {state.system_response}")
        else:
            st.info(f"**Agent:** {state.system_response}")

        st.subheader("📦 Yapılandırılmış Durum Çıktısı (JSON)")
        st.json(state.model_dump())

    with col_right:
        st.subheader("🌲 Sentaktik Bağımlılık Ağacı (Visual Dependency Tree)")
        # displacy SVG render
        svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 100})
        st.markdown(f'<div class="displacy-container">{svg}</div>', unsafe_allow_html=True)

    # 3. Dilbilimsel Token Analiz Tablosu (Tam Genişlik)
    st.markdown("---")
    st.subheader("🔍 Derin Dilbilimsel Sözcük İncelemesi (Token-Level Linguistic Features)")
    token_records = []
    for token in doc:
        token_records.append({
            "Token": token.text,
            "Lemma": token.lemma_,
            "POS (Part of Speech)": token.pos_,
            "Fine-grained Tag": token.tag_,
            "Syntactic Dependency": token.dep_,
            "Head Token (Ebeveyn)": token.head.text,
            "Children": [child.text for child in token.children]
        })
    st.dataframe(token_records, use_container_width=True)
