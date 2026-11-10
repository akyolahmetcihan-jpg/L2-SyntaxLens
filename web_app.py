import streamlit as st
import spacy
from pydantic import BaseModel
from typing import List, Optional

# Sayfa yapılandırması
st.set_page_config(page_title="LinguaOrder AI", page_icon="🍦", layout="wide")

# Dil modelini yükleme (requirements.txt ile önceden kurulmuş olacak)
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# Veri modelleri
class OrderItem(BaseModel):
    item: str
    quantity: int = 1
    customization: Optional[str] = None

class ParsedOrder(BaseModel):
    raw_text: str
    items: List[OrderItem] = []
    missing_information: List[str] = []
    status: str = "PENDING"

def extract_order_entities(user_text: str):
    doc = nlp(user_text.lower())
    parsed = ParsedOrder(raw_text=user_text)
    
    valid_flavors = ["vanilla", "chocolate", "strawberry", "mango"]
    valid_containers = ["cone", "cup"]
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    
    for token in doc:
        if token.text in valid_flavors:
            quantity = 1
            customization = None
            
            # Sayı tespiti (çocuk ve üst düğümlerden)
            candidate_tokens = list(token.children) + list(token.head.children)
            for cand in candidate_tokens:
                if cand.pos_ == "NUM" or cand.dep_ == "nummod":
                    if cand.text.isdigit():
                        quantity = int(cand.text)
                    else:
                        quantity = word_to_num.get(cand.text, 1)
                    break
            
            # Kap / Külah tespiti
            for container in valid_containers:
                if container in user_text.lower():
                    customization = container
            
            parsed.items.append(OrderItem(
                item=token.text,
                quantity=quantity,
                customization=customization
            ))
            
    if not parsed.items:
        parsed.missing_information.append("flavor_selection")
    else:
        for item in parsed.items:
            if not item.customization:
                parsed.missing_information.append(f"container_type_for_{item.item}")
                
    if not parsed.missing_information:
        parsed.status = "CONFIRMED"
        
    return parsed, doc

# --- STREAMLIT GÖRSEL ARAYÜZÜ ---
st.title("🍦 LinguaOrder AI")
st.markdown("**Rule-Grounded Conversational Agent & Syntactic Dependency Parser**")
st.caption("Computational Linguistics & Language Engineering Portfolio Demo")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Doğal Dil Girdisi (Utterance Input)")
    default_text = "I want two chocolate scoops in a cup please"
    user_input = st.text_input("Müşterinin Sipariş Cümlesi:", value=default_text)
    parse_button = st.button("Siparişi Çözümle (Parse)", type="primary")

if user_input or parse_button:
    result, doc = extract_order_entities(user_input)
    
    with col1:
        st.subheader("2. Durum ve Eksik Bilgi Kontrolü")
        if result.status == "CONFIRMED":
            st.success(f"Durum: {result.status} (Sipariş Hazırlanmaya Uygun)")
        else:
            st.warning(f"Durum: {result.status} (Eksik Parametre Var)")
            st.info(f"Tespit Edilen Eksikler: {', '.join(result.missing_information)}")

        st.subheader("3. Yapılandırılmış JSON Verisi (Backend Çıktısı)")
        st.json(result.model_dump())

    with col2:
        st.subheader("4. Dilbilimsel Sentaktik Analiz (Token & POS)")
        token_data = []
        for token in doc:
            token_data.append({
                "Token": token.text,
                "Lemma": token.lemma_,
                "POS (Rol)": token.pos_,
                "Dependency": token.dep_,
                "Head Noun": token.head.text
            })
        st.dataframe(token_data, use_container_width=True)

        st.subheader("5. Tespit Edilen Varlıklar (Entities)")
        if result.items:
            for item in result.items:
                st.markdown(f"- **Ürün:** `{item.item}` | **Adet:** `{item.quantity}` | **Format:** `{item.customization or 'Belirtilmedi'}`")
        else:
            st.write("Herhangi bir menü öğesi eşleşmedi.")
