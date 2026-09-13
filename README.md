# Smart Citizen Assistant

An AI-powered assistant that helps Egyptian citizens navigate government procedures — civil affairs, tax, and traffic — by answering questions in Arabic using verified legal sources, instead of requiring citizens to search scattered laws or visit multiple offices. Citizens can ask questions directly or upload a document/image for the system to read and analyze.

## How it works
Citizen Question / Uploaded Document
              │
              ▼
      OCR (if document/image)
              │
              ▼
         Router (MARBERT)
   classifies: civil_affairs / tax / traffic
              │
              ▼
       Hybrid RAG Retrieval
   Dense (FAISS + multilingual-E5) + Sparse (BM25)
              │
              ▼
        Generation LLM
   answers using only retrieved, relevant context
              │
              ▼
       Verification LLM
   checks the answer is actually supported by sources
              │
              ▼
          Final Answer

## Tech stack

| Layer | Technology |
|---|---|
| OCR | EasyOCR, PyMuPDF, `unstructured`, vision-LLM fallback (GPT-4o) for low-confidence scans |
| Text processing | Custom Arabic normalization/cleaning, chunking with overlap |
| Embeddings | `intfloat/multilingual-e5-small` (sentence-transformers) |
| Vector search | FAISS (dense) + BM25 (`rank_bm25`, sparse), fused via Reciprocal Rank Fusion |
| Routing | Fine-tuned MARBERT (`UBC-NLP/MARBERT`) classifier |
| Orchestration | LangGraph |
| Generation / Verification | OpenAI (GPT models) |
| Evaluation | Recall@K, Precision@K, MRR (retrieval), RAGAS (end-to-end) |

## Project structure

├── shared/                     # Common schemas/config used by every module
├── 01_ocr_person1/             # Document & image → text
├── 02_knowledge_base_person2/  # Legal corpus cleaning, chunking, metadata
├── 03_retrieval_person3/       # Embeddings + Hybrid RAG retrieval
├── 04_router_person4/          # Department classification (MARBERT)
├── 05_pipeline_eval_person5/   # LangGraph orchestration, generation, verification
├── integration/                # Full pipeline entry point + end-to-end test
├── evaluation_report/          # Combined metrics and findings
└── tests/                      # Per-module tests

## Setup

**1. Clone and create a virtual environment**
git clone https://github.com/Shahd132/Arabic-Government-Services-Assistant.git
cd Arabic-Government-Services-Assistant
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux


**2. Install dependencies**
pip install -r requirements.txt

**3. Set your API key**

Create a `.env` file in the project root:
OPENAI_API_KEY=your-key-here


## Running the project

**3. Run The model of fine tuning**
cd Arabic-Government-Services-Assistant/Arabic-Government-Services-Assistant\04_router_person4
python prepare_data.py
python train_router.py

**4. Run the full pipeline**:
cd Arabic-Government-Services-Assistant/web_app
python app.py

