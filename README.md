# L2-SyntaxLens

**A rule-based diagnostic workbench for morphosyntactic errors in L2 English.**

L2-SyntaxLens parses learner sentences with spaCy, reads the Universal Dependencies (UD) analysis, and applies a set of deterministic linguistic rules to flag errors. Each error comes with a target reconstruction and a short linguistic rationale. Results are validated with Pydantic v2 and can be exported as JSON.

## Features

- **Deterministic diagnostics:** the rules are explicit and inspectable, with no black-box scoring.
- **Rule trace:** every check and violation is logged in order.
- **Dependency tree:** an interactive displaCy SVG with custom dark styling.
- **Token matrix:** lemma, UPOS, XPOS, dependency relation, head and morphology for each token.
- **Typed export:** `SyntaxDiagnosticReport` serialized as JSON.
- **Benchmark presets:** one example sentence per rule, plus a well-formed control.

## Covered Phenomena

| Category | Example error | Suggested fix |
|---|---|---|
| Subject-verb agreement | *He go to school* | *He goes* |
| Prepositional collocation | *depend of* | *depend on* |
| Tense-aspect clash | *She writes an essay yesterday* | past tense (VBD) |
| Auxiliary double marking | *does not has* | *does have* |
| Determiner-noun number | *These book* | *These books* |
| Double negation | *didn't write nothing* | *didn't write anything* |
| Mass noun / quantifier | *many informations* | *much information* |
| Adjective ordering | *red big car* | *big red car* |
| Modal complementation | *can to go*, *must goes* | *can go*, *must go* |

## Installation

```bash
git clone <your-repo-url>
cd L2-SyntaxLens

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install streamlit spacy pydantic pandas
python -m spacy download en_core_web_sm
```

## Usage

```bash
streamlit run web_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`). Type a sentence, or pick a preset from the sidebar.

## Interface

| Tab | Content |
|---|---|
| Diagnostic Overview | Violation cards with focal violation, target reconstruction and rationale |
| Universal Dependencies Tree (SVG) | Dependency parse graph |
| Morphosyntactic Token Matrix (DataFrame) | Token-level attributes |
| Deterministic Rule Trace | Ordered log of rule checks |
| Export Pydantic Schema (JSON) | Validated report with download |

## Output Schema

```python
class DiagnosticError(BaseModel):
    category: str
    error_type: str
    token_focus: str
    suggested_correction: str
    linguistic_rule: str

class SyntaxDiagnosticReport(BaseModel):
    raw_utterance: str
    status: str                      # GRAMMATICAL | ERRORS_DETECTED
    error_count: int
    diagnostics: list[DiagnosticError]
    execution_trace: list[str]
    pedagogical_feedback: str
```

Example:

```json
{
  "raw_utterance": "She bought a red big car.",
  "status": "ERRORS_DETECTED",
  "error_count": 1,
  "diagnostics": [
    {
      "category": "Adjective Ordering",
      "error_type": "Prenominal Adjective Order Violation",
      "token_focus": "red big car",
      "suggested_correction": "big red car",
      "linguistic_rule": "Prenominal modifiers follow the canonical order: opinion < size < age < shape < colour < origin < material < purpose."
    }
  ]
}
```

## Project Structure

```
.
├── web_app.py     # Streamlit UI, rule engine and Pydantic schemas
└── README.md
```

## Extending the Rules

Rules live in `diagnose_l2_syntax()` and share one helper, `flag(...)`. To add a rule:

1. Add any lexical resources (word lists, mappings) near the top of the file.
2. Add a block in `diagnose_l2_syntax()` that inspects `doc` and calls `flag(category, error_type, focus, correction, rule, trace_msg)`.
3. Register a pill style in `CATEGORY_PILL`, and optionally add a preset to `TEST_CORPORA`.

## Limitations

- The rules are heuristic and depend on the `en_core_web_sm` parse, so a wrong parse can cause missed errors or false positives.
- The lexical resources (collocations, mass nouns, adjective classes) are small, curated lists, not exhaustive lexicons.
- Single sentences are the intended input; multi-sentence discourse is not modeled.

## Tech Stack

[spaCy](https://spacy.io) · [Universal Dependencies](https://universaldependencies.org) · [Pydantic v2](https://docs.pydantic.dev) · [Streamlit](https://streamlit.io) · pandas
