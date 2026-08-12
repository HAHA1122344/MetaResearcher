# MetaResearcher

A novel framework for scaling deep research agent training across four synergistic dimensions.

## Overview

MetaResearcher extends the LiteResearcher infrastructure with:

1. **Evolving Virtual World** - Injects temporal dynamics and adversarial misinformation
2. **Discovery-Oriented Tasks** - Beyond fact retrieval to hypothesis generation
3. **Self-Reflective Meta-Reward** - Multi-component reward in GRPO framework
4. **Heterogeneous Multi-Agent Swarm** - Scout/Filter/Synthesizer architecture

## Experimental Results

| Experiment | Dataset | Result |
|------------|---------|--------|
| E1: Single Agent | GAIA (100Q) | 80.0% accuracy |
| E4: 3-Agent Swarm | GAIA (100Q) | 82.0% accuracy |
| E5: Adversarial Robustness | 5 tasks | 0.0% drop |
| E6: Task Workflows | 5 complex tasks | 80% coverage |

## Repository Structure

```
MetaResearcher/
├── code/                    # Python source code
│   ├── src/                 # Core modules
│   │   ├── agent.py         # MetaResearcher agent
│   │   ├── benchmark.py     # GAIA benchmark
│   │   ├── meta_reward.py   # Meta-reward computation
│   │   ├── evolving_world.py # Adversarial environment
│   │   └── utils.py         # Utilities
│   ├── configs/             # Configuration files
│   └── notebooks/           # Colab notebooks
├── data/                    # Datasets
│   ├── gaia_mini.jsonl      # 30-question GAIA subset
│   └── adversarial_corpus.jsonl  # Adversarial scenarios
├── results/                 # Experiment results
├── figures/                 # Generated figures
├── MetaResearcher.tex       # LaTeX source (paper)
├── MetaResearcher.bib       # References
└── MetaResearcher.pdf       # Generated PDF
```

## Requirements

- Python 3.10+
- openai
- transformers
- torch
- accelerate

## Usage

### Run Experiments

```bash
# Set API key
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://your_api/v1"

# Run experiments
python code/run_experiment.py
```

### Colab Setup

See `code/notebooks/colab_setup.ipynb` for Google Colab instructions.

## Citation

```bibtex
@article{yu2026metaresearcher,
  title={MetaResearcher: A Framework for Scaling Deep Research Agent Training},
  author={Yu, Wei and Liu, Suxing and Yu, Minjie and Wang, Jiahao and Zheng, Zhijian and Deng, Haocheng and Li, Bing},
  journal={Electronics},
  year={2026}
}
```

## License

MIT License
