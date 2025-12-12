# NGBSS Guide Processing Module

This module extracts procedural steps from NGBSS PDF guides using **Gemini Flash Vision** (FREE) and stores them in ChromaDB for RAG retrieval.

## 🏆 Hackathon Bonus Feature

This implements the "Bonus spécial" from the hackathon:
> *"Un bonus sera accordé aux équipes qui exploitent efficacement l'OCR pour extraire les informations présentes uniquement dans ces images."*

## 📁 Folder Structure

```
src/ngbss/
├── __init__.py          # Module init
├── gemini_vision.py     # Gemini Flash API with key rotation
├── ingest_ngbss.py      # Main ingestion pipeline
└── README.md            # This file

data/ngbss/              # Put your NGBSS PDF files here!
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install google-generativeai PyMuPDF Pillow
```

### 2. Add Your NGBSS PDF Files

Copy your 15 NGBSS PDF files to:
```
data/ngbss/
├── Guide NGBSS Recharge par bon de commande.pdf
├── Guide NGBSS Création nouveau Pack IDOOM Fibre.pdf
├── ... (other PDF files)
```

### 3. Run the Ingestion

```bash
python -m src.ngbss.ingest_ngbss --pdf-folder data/ngbss
```

### 4. Verify the Output

The pipeline will:
1. ✅ Extract each PDF page as an image
2. ✅ Send to Gemini Flash for structured extraction
3. ✅ Save extracted data to `data/ngbss_extracted.json`
4. ✅ Ingest into ChromaDB collection `ngbss-guides`

## 🔧 Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--pdf-folder` | `data/ngbss` | Path to NGBSS PDF files |
| `--chroma-dir` | `data/chroma_db` | ChromaDB persistence directory |
| `--collection` | `ngbss-guides` | ChromaDB collection name |
| `--no-json` | False | Skip saving to JSON |

## 🔑 API Keys

The module uses **37 Gemini API keys** with automatic rotation to avoid rate limits. The keys are stored in `gemini_vision.py`.

### Free Tier Limits (per key)
- 15 requests/minute
- 1,500 requests/day
- Vision included!

With 37 keys × 15 RPM = **555 requests/minute** capacity!

## 📊 Output Format

Each extracted step is stored with this metadata:

```json
{
    "page_content": "Procédure: Recharge par Bon de Commande\nÉtape 3: Sélectionner le compte...",
    "metadata": {
        "source": "Guide NGBSS Recharge.pdf",
        "procedure_name": "Recharge par Bon de Commande",
        "step_order": 3,
        "total_steps": 8,
        "type": "ngbss_guide",
        "action": "Sélectionner le compte de paiement",
        "navigation": "Facturation > Paiement"
    }
}
```

## 🎯 Integration with Main Chat

After ingestion, update your chat.py to query BOTH collections:

```python
# Query offers collection
offers_retriever = offers_vectorstore.as_retriever(search_kwargs={"k": 5})

# Query NGBSS guides collection
ngbss_vectorstore = Chroma(
    collection_name="ngbss-guides",
    embedding_function=embeddings,
    client=chroma_client
)
ngbss_retriever = ngbss_vectorstore.as_retriever(search_kwargs={"k": 5})

# Intent detection to choose which retriever to use
if "NGBSS" in query or "étapes" in query or "comment faire" in query:
    docs = ngbss_retriever.invoke(query)
else:
    docs = offers_retriever.invoke(query)
```

## 🧪 Testing

Test Gemini connection:
```bash
python -m src.ngbss.gemini_vision
```

Expected output:
```
🔑 Using API key #1
✅ Gemini Flash is working!
```
