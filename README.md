# CI Log Triage Copilot

This project is a CI Log Triage Copilot designed to accelerate the diagnosis of failures in Continuous Integration/Continuous Deployment (CI/CD) pipelines. It uses a lightweight agent that combines rule-based heuristics with Retrieval-Augmented Generation (RAG) to classify failures, retrieve similar incidents from a knowledge base, and recommend actionable fixes.

## Features

- **Rule-Based Weak Labeling:** Quickly classifies common failures using a set of predefined regex rules.
- **Hybrid Retrieval:** Combines dense (FAISS) and sparse (BM25) search for accurate, context-aware retrieval from a knowledge base.
- **Lightweight Agent:** A `propose -> verify -> refine` loop that uses simulated "tools" to verify findings without requiring a live LLM.
- **Structured Output:** Generates a "Triage Card" for each failure with a likely root cause, suggested actions, and evidence-based citations.
- **Evaluation Mode:** Includes a script to evaluate the agent's performance against a gold-standard dataset.

## Project Structure

The project is organized into a modular structure for clarity and maintainability:

- `ci_log_triage_copilot.py`: The main entry point for running the application.
- `evaluate.py`: A script to run performance evaluations for the retrieval and triage components.
- `config.py`: Stores all global configurations, paths, and constants.
- `requirements.txt`: Lists all Python dependencies.
- `src/`:
    - `agent.py`: Contains the core agent logic, tools, and retrieval functions.
    - `data_utils.py`: Handles data loading and preprocessing.
    - `kb_builder.py`: Manages the creation of the knowledge base and search indexes.
- `output/`: The default directory for all generated artifacts (indexes, KB files, reports).

## Setup

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

The first time you run the script, it will automatically create the knowledge base and build the necessary search indexes in the `output/` directory.

### Running on Log Data

To triage failures from a CSV file, provide the path to the file. Use the `--limit` flag to control how many unique failures are processed.

```bash
python ci_log_triage_copilot.py ci_cd_logs.csv --limit 5
```

To force a rebuild of the knowledge base and indexes, use the `--build` flag:
```bash
python ci_log_triage_copilot.py ci_cd_logs.csv --build
```

### Running the Demonstration

To see the agent analyze a diverse, hardcoded set of interesting log messages, run the script with the `--demo` flag.

```bash
python ci_log_triage_copilot.py --demo
```

### Running Evaluation

To evaluate the performance of the agent against the gold-standard data, run the `evaluate.py` script.

```bash
python evaluate.py
```