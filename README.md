# Exam to AI Benchmark

**Data Science Lab Fall 2025 @ ETH Zurich**

A pipeline for converting PDF exams into structured AI benchmarks, evaluating LLM performance, and visualizing results.

## Key Features

*   **PDF to Benchmark**: Automated conversion of raw exam and solution PDFs into structured JSON datasets using Multimodal LLMs.
*   **Benchmarking**: Evaluate any LLM on processed exams and automatically grade answers against official solutions using an LLM-as-a-Judge system.
*   **RAG Support**: Integrated Retrieval-Augmented Generation pipeline to evaluate models with access to external reference documents.
*   **Visualization & Analysis**: Tools to aggregate results, calculate statistics, and generate comparison plots.

## Setup

### 1. Install Dependencies

Ensure you have Python 3.9+ installed. (tested on python 3.9.19)

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory with your API keys:

```env
# OpenRouter API Configuration (for most models)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_key_here

# Swiss AI / CSCS API Configuration (for Apertus models)
SWISSAI_API_KEY=your_swissai_key_here
```

### 3. Project Configuration

All configuartions are done in `config/config.yaml`, where detailed desctiptions of each parameter are given. Key sections include:

*   **Paths**: Set `raw_data_dir`, `processed_data_dir`, etc.
*   **Processing**: Define which professions and exam numbers to process.
*   **Benchmarking**: List models to test in standard mode.
*   **Benchmarking RAG**: List models to test with RAG, and configure RAG parameters (chunk size, top_k).
*   **Evaluation**: List all models (standard and RAG) to include in the final plots.

### 4. Data Setup

Organize your raw exam files in the `data/raw` directory. Use the following structure:

```
data/raw/
└── {Profession Name}/
    ├── 1/                  # Exam Number 1
    │   ├── exam.pdf        # The exam questions
    │   └── solution.pdf    # The official solution
    ├── 2/                  # Exam Number 2
    │   ├── exam.pdf
    │   └── solution.pdf
    └── ...
```

*   **Profession Name**: The name of the profession. We use this naming convention because our project focused on using Swiss Federal Professional Exams, but this could be any domain like "math" or "physics". This names are used in the `professions` list in `config.yaml`.
*   **Exam Number**: A sequential number or identifier for the exam (e.g., "1", "2"). This must match the `exam_numbers` list in `config.yaml`.
*   **Files**: Each folder must contain exactly one `exam.pdf` and one `solution.pdf`.

## Usage

The pipeline consists of four main stages:

### 1. Data Preprocessing

Converts raw PDF exams and solutions into structured JSON files (`answer_sheet.json`, `solution_sheet.json`).

```bash
python processing/processing_main.py
```

*   **Input**: PDFs in `data/raw/{profession}/{exam_number}/`
*   **Output**: JSONs in `data/processed/{profession}/{exam_number}/{timestamp}/`

### 2. RAG Database Creation (Optional)

If you plan to use RAG, you must first ingest the reference documents into the vector database.

```bash
python RAG/create_rag_db.py
```

*   **Input**: Reference documents in `data/rag/documents/{profession}/`.
*   **Output**: ChromaDB vector store in `data/rag/vector_database/{profession}/`.

### 3. Benchmarking

#### Standard Benchmarking
Runs models without external context.

```bash
python benchmarking/benchmarking_main.py
```

#### RAG Benchmarking
Runs models with context retrieved from the vector database.

```bash
python benchmarking/benchmarking_rag_main.py
```

*   **Note**: This script automatically appends `_rag` to the output folder name (e.g., `model=gpt-4_rag`). You do **not** need to add `_rag` to the model name in the `benchmarking_rag` config section; just use the base model name.

### 4. Evaluation

Aggregates results from all runs and generates comparison plots.

```bash
python eval/eval_main.py
```

*   **Configuration**: In the `evaluation` section of `config.yaml`, add `_rag` to the model names you want to plot (e.g., `openai/gpt-4` and `openai/gpt-4_rag`) to distinguish between the two runs.
*   **Output**: Plots and metadata in `data/eval/{timestamp}/`

## Data Structure

```
data/
├── raw/
│   └── {profession}/
│       └── {exam_number}/
│           ├── exam.pdf
│           └── solution.pdf
├── processed/
│   └── {profession}/
│       └── {exam_number}/
│           └── {timestamp}/
│               ├── answer_sheet.json
│               └── solution_sheet.json
├── rag/
│   ├── documents/
│   │   └── {profession}/         # Reference Markdown/PDF files
│   └── vector_database/
│       └── {profession}/         # ChromaDB files
├── benchmarked/
│   └── {profession}/
│       └── {exam_number}/
│           └── {processing_timestamp}/
│               ├── model={model_name}/
│               │   └── {benchmark_timestamp}/
│               │       ├── model_answers.json
│               │       └── judge={judge_name}/
│               │           └── graded_answers.json
│               └── model={model_name}_rag/  # Created by benchmarking_rag_main.py
│                   └── ...
└── eval/
    └── {timestamp}/
        ├── model_comparison_{timestamp}.png
        └── evaluation_metadata.json
```
