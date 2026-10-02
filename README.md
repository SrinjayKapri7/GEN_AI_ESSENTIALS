# Generative AI Practice

A hands-on collection of Jupyter notebooks for learning generative AI, from text encoding and embeddings to fine-tuning, retrieval-augmented generation (RAG), and agent workflows with LangGraph.

The numbered folders form a learning path. Each topic combines explanations with practical experiments, and several folders include sample documents, generated parsing outputs, or saved vector indexes.

Some additional fine-tuning and embedding code is available in Google Colab.

## Learning path

| Module | Topics |
| --- | --- |
| [01 - Encoding and embeddings](01-encoding-and-embedding-introduction/) | Introduction to representing text for machine learning |
| [02 - Text encoding](02-text-encoding/) | One-hot encoding, bag of words, and TF-IDF |
| [03 - Word2Vec](03-word2vec-embeddings/) | Pretrained word vectors and training custom embeddings |
| [04 - LLM embeddings](04-llm-embeddings/) | Sentence Transformers, hosted embeddings, and CLIP |
| [05 - LLMs and multimodal models](05-llm-and-multimodal-models/) | Accessing models and working with multimodal inputs |
| [06 - Prompt engineering](06-prompt-engineering/) | Prompt templates, chat prompts, and dynamic prompting |
| [07 - Fine-tuning](07-fine-tuning/) | Hugging Face and Unsloth workflows, LoRA/QLoRA, instruction tuning, preference tuning, and vision models |
| [08 - RAG data parsing](08-rag-data-parsing/) | LangChain loaders, PyMuPDF, pdfplumber, Docling, LlamaParse, and a RAG pipeline |
| [09 - RAG chunking](09-rag-chunking/) | Fixed-size, recursive, token, structure-aware, semantic, and parent-child chunking |
| [10 - Vector databases](10-rag-vector-databases/) | FAISS, Chroma, Pinecone, and Qdrant |
| [11 - Retrieval](11-rag-retrieval/) | Similarity search, BM25, MMR, hybrid retrieval, query expansion, compression, and parent-document retrieval |
| [12 - Agentic AI](12-agentic-ai-langgraph/) | LangGraph state, messages, routing, and tool-based workflows |

## Getting started

### 1. Create a Python environment

The repository specifies Python **3.13** in `.python-version` and Python **>=3.13** in `pyproject.toml`. Some fine-tuning tools have their own Python, CUDA, and platform requirements; use a separate compatible environment for those notebooks when needed.

From the repository root, on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab
```

`requirements.txt` contains the shared dependencies. The dependency list in `pyproject.toml` is currently empty, so installing the project alone does not install the notebook dependencies.

### 2. Configure credentials as needed

Create a `.env` file at the repository root and add only the credentials required by the notebook you want to run. For example:

```dotenv
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_hugging_face_token
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_Cluster_Endpoint=your_qdrant_cluster_url
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
```

The Qdrant endpoint variable above preserves the spelling used in the notebook. Some integrations, such as search tools or LangSmith, may require additional configuration in their cells.

Many notebooks use `python-dotenv` to load credentials. If your selected notebook does not, run this before its API calls:

```python
from dotenv import load_dotenv

load_dotenv()
```

`.env` is already ignored by Git. Keep credentials out of notebook cells and saved outputs. Hosted model calls and cloud services may incur charges; local encoding and Word2Vec exercises are useful starting points without API credentials.

### 3. Open a notebook

```bash
python -m jupyterlab
```

Open a notebook and choose the kernel from your new environment. For VS Code or another notebook editor, register it explicitly if needed:

```bash
python -m ipykernel install --user --name gen-ai-practice --display-name "Python (Gen AI Practice)"
```

Run cells from top to bottom, starting with the imports and configuration. The learning material lives in the notebooks; `main.py` is a placeholder that prints a greeting.

## Notebook-specific requirements

The shared requirements do not cover every module. Check the imports and installation cells in your chosen notebook before running it. Examples of additional packages used in the repository include:

| Area | Additional packages or setup |
| --- | --- |
| Chunking | `langchain-text-splitters`, `tiktoken`, `matplotlib`; NLTK data downloads where requested |
| Local vector stores | `faiss-cpu` or `langchain-chroma` / `chromadb`, depending on the notebook |
| Hosted vector stores | `pinecone` / `langchain-pinecone`, or `qdrant-client` / `langchain-qdrant`, plus service configuration |
| Advanced retrieval | `langchain-classic` for notebooks using its retriever imports |
| Document parsing | `docling`, `pytesseract`, or `llama-cloud`; OCR examples also need the Tesseract executable |
| Fine-tuning | Notebook-specific versions of `unsloth`, `trl`, `peft`, `accelerate`, and quantization libraries; suitable GPU resources |
| Agent workflows | `langgraph`, `langchain-tavily`, and the model/vector-store integrations imported by the notebook |

Install these in the selected kernel's environment. Preserve explicit version pins in fine-tuning notebooks rather than installing every optional package into one environment.

## Working with data and models

- Review document paths before running cells. Some examples use absolute paths from the author's machine or Colab; replace them with paths to your own files or the included samples.
- Check the notebook's working directory when resolving relative paths. Data and output folders often sit beside the notebook.
- Model weights, embedding models, and datasets may download on first use. Some Hugging Face models require accepting their access terms and providing a token.
- Fine-tuning notebooks need appropriate compute and input datasets. CPU-friendly preprocessing exercises can be explored separately from training cells.
- Saved FAISS and Chroma indexes are included in some folders. Rebuild them when changing the source documents, embedding model, or chunking settings.
- Review cells that create or delete cloud indexes, rebuild local stores, or upload trained models before executing them.

## Suggested first session

1. Explore [text encoding](02-text-encoding/encoding.ipynb) to compare one-hot, bag-of-words, and TF-IDF representations.
2. Try [Word2Vec](03-word2vec-embeddings/w2vec.ipynb) and inspect similarities between word vectors.
3. Work through [chunking methods](09-rag-chunking/Chunking_Methods.ipynb) to see how document boundaries affect retrieval.
4. Build a small RAG experiment with [FAISS](10-rag-vector-databases/FAISS/FAISS.ipynb), then explore the [retrieval notebooks](11-rag-retrieval/).
5. Continue to [LangGraph](12-agentic-ai-langgraph/Langraph_introduction.ipynb) or the [fine-tuning workflows](07-fine-tuning/) once the foundations are familiar.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Install the missing notebook-specific dependency into the active kernel, then restart the kernel. |
| API authentication error | Confirm the required variable is loaded and the account has access to the selected model or service. |
| File not found | Replace machine-specific paths and check the notebook's working directory. |
| GPU memory error | Reduce batch size or sequence length, or use the notebook's quantization settings and a suitable GPU. |
| Import or version mismatch | Check notebook installation cells and version pins; isolate conflicting workflows in separate environments. |

## Contributing

Add examples to the matching topic folder, explain the experiment and its prerequisites, and use portable paths where possible. Before sharing a notebook, remove credentials and private data from both cells and outputs, and document any extra dependencies or compute requirements.
