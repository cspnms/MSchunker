# MSchunker  
### Smart Text Chunker for LLM Preprocessing

Smart text chunker for LLM preprocessing (sections → paragraphs → sentences → hard splits).

---

## Features

- **Hierarchical splitting pipeline**
  - Detects sections (Markdown headers / underlined titles)
  - Splits into paragraphs
  - Splits into sentences
  - Falls back to hard token/character splitting when needed

- **Controls for chunk size**
  - `max_tokens`
  - `max_chars` (hard limit)
  - Approx token counting or custom tokenizer

- **Overlap support**
  - Preserves context between chunks
  - Deterministic and clean

- **Rich metadata per chunk**
  - Section index / heading  
  - Paragraph index  
  - Character offsets  
  - Split reason (paragraph boundary, sentence limit, hard limit)  
  - Chunk index  

- **Zero dependencies**
  - Pure Python
  - Semantic/embedding modes can be added later as extensions

---

## Installation (development mode)

```bash
git clone https://github.com/cspnms/MSchunker.git
cd MSchunker
pip install -e .
