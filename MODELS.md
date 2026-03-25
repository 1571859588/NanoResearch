# Local Pretrained Models

This document lists all local pretrained models available for use in NanoResearch experiments.

## Model Index

| Name | Location | Type | Description |
|------|----------|------|-------------|
| ESM-2 | `local_models/ESM-2/` | Protein language model | Meta's Evolutionary Scale Modeling |
| ProtBERT | `local_models/ProtBERT/` | Protein language model | BERT-based model for protein sequences |

---

## ESM-2 (Evolutionary Scale Modeling 2)

**Purpose**: Protein sequence representation, protein function prediction, protein design

**Location**: `local_models/ESM-2/esm2_t33_650M_UR50D/`

### Model Overview

- **650M parameters**
- Trained on 2.1B protein sequences from UniRef50
- Captures evolutionary patterns in protein sequences
- Outputs contextualized protein embeddings

### Usage in NanoResearch

```python
import torch
import esm

# Load model from local path
MODEL_PATH = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_models/ESM-2/esm2_t33_650M_UR50D")

model, alphabet = esm.pretrained.model_from_path_or_url(str(MODEL_PATH))
model = model.eval()

# Get protein embeddings
batch_labels, batch_tokens = alphabet.to_batch(["MKTITALLIVIVIVIMAVTTTTT..."])
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    embeddings = results["representations"][33]
```

---

## ProtBERT

**Purpose**: Protein language understanding, protein function prediction

**Location**: `local_models/ProtBERT/protbert/`

### Model Overview

- Based on BERT architecture
- Trained on 1.5B protein sequences
- 12 layers, 768 hidden size
- Outputs contextualized protein embeddings

### Usage in NanoResearch

```python
from transformers import BertTokenizer, BertModel

MODEL_PATH = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_models/ProtBERT/protbert")

tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
model = BertModel.from_pretrained(str(MODEL_PATH))

# Get protein embeddings
tokens = tokenizer("MKTITALLIVIVIVIMAVTTTTT...", return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
```

---

## How to Add New Models

1. Place model files in `local_models/<model_name>/<model_type>/`
2. Create a `README.md` describing:
   - Model architecture
   - Training data
   - How to load the model
3. Add entry to this `MODELS.md` file

---

## Notes for Pipeline

When the pipeline runs:
1. Check `MODELS.md` to discover available local models
2. When the experiment blueprint requests a pretrained model, first check if it exists locally
3. If found locally, symlink or copy to workspace instead of downloading from HuggingFace/ModelScope
4. Provide model metadata (name, type, embedding dimension) to the CODING stage
