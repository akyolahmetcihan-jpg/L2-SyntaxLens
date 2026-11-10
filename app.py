import spacy
from pydantic import BaseModel
from typing import List, Optional

nlp = spacy.load("en_core_web_sm")

class OrderItem(BaseModel):
    item: str
    quantity: int = 1
    customization: Optional[str] = None

class ParsedOrder(BaseModel):
    raw_text: str
    items: List[OrderItem] = []
    missing_information: List[str] = []
    status: str = "PENDING"

def extract_order_entities(user_text: str) -> ParsedOrder:
    doc = nlp(user_text.lower())
    parsed = ParsedOrder(raw_text=user_text)
    
    valid_flavors = ["vanilla", "chocolate", "strawberry", "mango"]
    valid_containers = ["cone", "cup"]
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    
    for token in doc:
        if token.text in valid_flavors:
            quantity = 1
            customization = None
            
            # Sayı tespiti (kendi bağlı olduğu kelimeye ve üst kelimeye bakar)
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
        
    return parsed

# Test cümlesi
test_input = "I want two chocolate scoops in a cup please"
result = extract_order_entities(test_input)

print("\n================== ANALİZ SONUCU ==================")
print(f"Kullanıcı Cümlesi : {test_input}")
print(f"Sipariş Durumu    : {result.status}")
print("Çıkarılan Bilgiler:")
print(result.model_dump_json(indent=2))
print("===================================================\n")
