# L2-SyntaxLens: Pedagogical Dependency & Morphosyntactic Error Diagnostic Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lingua-order-agent-eqyrq9frcuablyzhqczaqx.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/NLP-spaCy%20v3.8-09A3D5.svg)](https://spacy.io/)
[![Schema](https://img.shields.io/badge/Validation-Pydantic%20v2-e92063.svg)](https://pydantic.dev/)

An explainable, deterministic natural language parsing workbench designed to diagnose morphosyntactic, concord, and collocational anomalies in second-language (L2) English learner corpora. 

Built on top of formal **Universal Dependencies (UD)** tree-traversal algorithms and **Pydantic** data contracts, **L2-SyntaxLens** bridges theoretical syntax, pedagogical grammar, and computational language engineering.

---

##  Theoretical Foundations & Architecture

Standard end-to-end Large Language Models (LLMs) used for grammatical error correction often suffer from non-deterministic behavior, opacity, and token hallucinations. **L2-SyntaxLens** implements a rule-grounded, inspectable diagnostic pipeline that maps surface structures to syntactic trees and evaluates structural well-formedness through formal constraints:
Raw L2 Learner Utterance
                         |
                         v
          [spaCy Dependency Parsing Engine]
                         |
          |-- Tokenization and Lemmatization
          |-- Morphosyntactic Feature Extraction (UPOS / XPOS)
          `-- Syntactic Dependency Graph (Universal Dependencies)
                         |
                         v
     [Formal Constraint and Diagnostic Rule Traversal]
          |-- Subject-Verb Concord (nsubj -- head predicate)
          |-- Subcategorization and Prepositional Complementation
          |-- Temporal Deictic Anchor vs Aspectual Morphology
          |-- Auxiliary Complementation (Do-Support and Modals)
          |-- NP Internal Agreement (Determiner-Head Concord)
          `-- Polarity Scope and Double Negation Isolation
                         |
                         v
             [Pydantic Diagnostic Schema]
                         |
    +--------------------+--------------------+
    v                                         v
Pedagogical Feedback and Correction      Interactive Visual Workbench
(Linguistic Rule Rationale)            (SVG Dependency Graph + POS Matrix)
## Modeled Linguistic Phenomena

| Error Classification | Target Linguistic Rule | Structural Mechanism and UD Relation |
| :--- | :--- | :--- |
| Subject-Verb Concord | Third-person singular present concord (-s inflection) | Resolves nsubj agreement with governing verb head (VBZ vs VBP). |
| Lexicosyntactic Collocation | Subcategorization frame selection for prepositional predicates | Verifies prep and adp complement nodes against target lexical heads. |
| Tense-Aspect Clash | Strict past deictic constraint on finite verb morphology | Flags present finite verbs under the scope of past temporal adverbials (advmod, pobj). |
| Auxiliary Redundancy | Inflection absorption by do-support auxiliaries | Prevents double tense and agreement marking on auxiliary complements. |
| NP Internal Concord | Number concordance between determiners and count nouns | Inspects det number features against head noun morphology (NN vs NNS). |
| Negative Polarity | Standard English single clausal negation | Detects redundant negative operators (neg) within local clausal domains. |
| Modal Syntax | Bare infinitival complement selection | Enforces uninflected lexical head complements following modal auxiliaries (MD). |

## Technical Specifications

- Programming Language: Python 3.10+
- Natural Language Processing: spaCy (en_core_web_sm pipeline)
- Schema Validation: Pydantic v2
- User Interface: Streamlit and displaCy SVG dependency renderer
- Inference Performance: Sub-15ms CPU execution time

## Local Deployment

1. Clone the repository:
```bash
git clone [https://github.com/akyolahmetcihan-jpg/lingua-order-agent.git](https://github.com/akyolahmetcihan-jpg/lingua-order-agent.git)
cd lingua-order-agent
