# LOCA: Logical Chain Augmentation for Scientific Corpus Cleaning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**LOCA (Logical Chain Augmentation)** is a novel framework for automatically cleaning scientific corpora, specifically designed to address the high error rates commonly found in scientific question-answering (QA) datasets.

## Installation

### Prerequisites
- Python 3.12+
- LLM API key
- [uv](https://docs.astral.sh/uv/) (recommended)

### Setup

1. **Clone the repository**
```bash
git clone xxx
cd LOCA
```

2. **Install dependencies**

Using uv (recommended):
```bash
uv sync
```

## Project Structure

```
LOCA/
├── src/                         # Source code
│   ├── loca/                    # LOCA implementation
│   │   ├── solver.py            # Main LOCA solver
│   │   ├── results/             # LOCA results
│   │   └── utils/               # Utilities
│   │       ├── augmentation.py  # Augmentation agent
│   │       ├── reviewer.py      # Review agents
│   │       └── secretary.py     # Secretary for summarizing reviews
│   ├── api/                     # LLM API interfaces
│   ...
├── configs/                     # Configuration files
├── problem_set/                 # Test datasets
├── test_results/                # Evaluation results on other methods
└── scripts/                     # Analysis scripts
```

## Usage
### Running LOCA
```bash
uv run main.py --config PATH_TO_YAML --config-name CONFIG_NAME_IN_YAML
```

### Automated External Consistency Check
```bash
./scripts/run_analyze_improved_solutions.sh
```
