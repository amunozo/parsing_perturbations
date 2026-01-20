# Another Dead End for Morphological Tags? Perturbed Inputs and Parsing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2023.findings-acl.459/)

This repository contains the official implementation for the paper:
**[Another Dead End for Morphological Tags? Perturbed Inputs and Parsing](https://aclanthology.org/2023.findings-acl.459/)**
*Alberto Muñoz-Ortiz and David Vilares*
Presented at **Findings of ACL 2023** in Toronto, Canada.

---

## Overview

This project investigates the impact of morphological information on the robustness of different parsing architectures (Transition-based, Graph-based, and Sequence Labeling) against lexical perturbations. It includes an adversarial attack framework to evaluate how parsers behave when faced with noisy or perturbed inputs.

## Project Structure

The repository is organized as follows:

- `src/`: Core logic for parsers and perturbation strategies.
  - `gb.py`: Graph-based parsing (SuPar biaffine).
  - `tb.py`: Transition-based parsing (Syntactic Pointer).
  - `sl.py`: Sequence Labeling parsing (dep2label).
  - `tagger.py`: NCRF++ based part-of-speech tagger.
  - `perturbate.py`: Logic for applying adversarial attacks.
  - `attacks.py`: Specific character-level attack implementations.
- `scripts/`: Entry-point scripts for training and evaluation.
- `notebooks/`: Jupyter notebooks for data analysis and result visualization.
- `results/`: Directory for storing logs and scores.
- `data/`: Placeholder for UD treebanks (UD 2.9 recommended).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amunozo/parsing_perturbations.git
   cd parsing_perturbations
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   The project relies on external libraries like `supar`, `transformers`, `nltk`, and `NCRF++`. Ensure these are installed according to the original requirements (if available) or by installing the submodules if present.

## Usage

### Training Graph-based Parsers
```bash
python scripts/gb_train.py [tags] [device]
```
Where `[tags]` can be `none`, `upos`, `xpos`, or `feats`, and `[device]` is the CUDA device ID.

### Evaluation with Perturbations
To evaluate a Graph-based parser under various levels of perturbation:
```bash
python scripts/gb_evaluate.py [device]
```

Similarly, you can use `sl_train.py`/`sl_evaluate.py` for sequence labeling models and `tb_train.py`/`tb_evaluate.py` for transition-based models.

## Citation

If you use this code or our findings in your research, please cite:

```bibtex
@inproceedings{munoz-ortiz-vilares-2023-another,
    title = "Another Dead End for Morphological Tags? Perturbed Inputs and Parsing",
    author = "Mu{\~n}oz-Ortiz, Alberto and
      Vilares, David",
    editor = "Rogers, Anna and
      Boyd-Graber, Jordan and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.459/",
    doi = "10.18653/v1/2023.findings-acl.459",
    pages = "7301--7310",
}
```

## Contact

For any questions or issues, please contact the main author:
**Alberto Muñoz-Ortiz** - [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Acknowledgments

This work was supported by the European Research Council (ERC), ERDF/MICINN-AEI, and Xunta de Galicia.
