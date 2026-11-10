import streamlit as st
import spacy
from spacy import displacy
from pydantic import BaseModel, Field
from typing import List, Optional
import time

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="LinguaOrder | Dependency Dialogue Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN VE AKADEMİK TEMA STİLLERİ ---
st.markdown("""
<style>
    /* Temiz Tipografi */
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* DisplaCy SVG Karanlık Mod Entegrasyonu */
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

    /* Durum Rozeti (Status Badge) */
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

# Dil Modelini Önbellekleme
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# --- VERİ VE ŞEMA TANIMLARI ---
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

# Dilbilimsel Sözlük Kuralları
VALID_FLAVORS = {
    "vanilla": ["vanilla"],
    "chocolate": ["chocolate", "choco"],
    "strawberry": ["strawberry"],
    "mango": ["mango"],
    "pistachio": ["pistachio"]
}
VALID_CONTAINERS = ["cone", "cup", "tub", "bowl"]
NUMBER_MAP = {
    "a": 1, "an": 1, "one": 1, "
