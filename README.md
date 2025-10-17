### Overview

![流程图](./images/overview.pdf)
This repository contains the implementation and example projects for the paper **Retrieval-Augmented PLC Code Generation with Static-Analysis Feedback**. It targets Siemens TIA Portal and focuses on automatic generation and evaluation of SCL (Structured Control Language) programs. The system retrieves relevant examples, builds prompts, calls a large language model to generate SCL code, and offers optional automation/testing scripts to reproduce results.

### Repository Structure
- **code/**: Main source code
  - `generator.py`: Entry point that retrieves examples, builds prompts, calls the LLM, and writes generated SCL
  - `static_feedback.py`: Stactic feedback modules
  - `score_fixed_example.py`: Reranker for rule based examples
  - `BM25Retriever.py:` BM25 retriever
- **data/**: Dataset
  - `samples_clean.jsonl`: 
  -  `questions.jsonl`: Question dataset
  - `fixed_examples.jsonl`: Rule based examples
- **TestProject.zap19**: Evaluation project for TIA Portal

### Core Idea (Brief)
1. Use BM25 and dense embeddings to retrieve the most relevant SCL examples;
2. Construct prompts combining code templates and reference examples;
3. Invoke a causal LLM to generate target SCL code (filling the 【】 blocks);
4. Write results to `res/` as `.scl` files;
5. Optionally validate by compiling and running tests via the provided TIA scripts.

### Quick Start (Generate SCL)
After configuring paths, run:

```powershell
python ./code/generator.py
```

The program will generate SCL code.

For static feedback, run:

```powershell
python ./code/static_feedback.py
```

If you want to perform the evaluation, please import the `TestProject.zap19` file into the TIA Portal platform and run the test program.
