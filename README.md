# VSFC-VSMEC-VTOC-VMNLI

This repository contains scripts and datasets for training Vietnamese text classification models on ViGLUE tasks using Qwen2.5.

## 📚 Tasks Included

| Task    | Description                                 | Labels                             |
|---------|---------------------------------------------|------------------------------------|
| VSFC    | Sentiment of student feedback               | positive, neutral, negative        |
| VSMEC   | Emotions from social media                  | enjoyment, anger, fear, etc.       |
| VTOC    | Vietnamese news topic classification        | Law, Travel, World, etc. (15 total)|
| VMNLI   | Natural Language Inference (premise-hypo)   | entailment, neutral, contradiction |

---

## 📁 Project Structure

```
VSFC/
│   ├── preprocess_VSFC.py
│   ├── train_VSFC.py
│   └── demo_VSFC.py
VSMEC/
│   ├── preprocess_VSMEC.py
│   ├── train_VSMEC.py
│   └── demo_VSMEC.py
VTOC/
│   ├── preprocess_VTOC.py
│   ├── train_VTOC.py
│   └── demo_VTOC.py
VMNLI/
│   ├── preprocess_VMNLI.py
│   ├── train_VMNLI.py
│   └── demo_VMNLI.py
VieGLUE/
└── data/
    ├── ax/
    ├── cola/
    ├── mnli/        ← VMNLI data
    ├── mrpc/
    ├── qnli/
    ├── qqp/
    ├── rte/
    ├── sst2/
    ├── stsb/
    ├── vnrte/
    ├── vsfc/
    ├── vsmec/
    ├── vtoc/
    └── wnli/
extract_vigluedata.py
check_ready_tasks.py
.gitattributes
.gitignore
README.md

```

---

## ⚙️ Scripts

- `extract_vigluedata.py`: Extracts and renames `.tar.gz` files (e.g. `dev.json` → `validation.json`)
- `check_ready_tasks.py`: Checks which task folders are ready for `load_dataset()`
- `preprocess_*.py`: Prepares and tokenizes datasets
- `train_*.py`: Trains model using Hugging Face `Trainer`
- `demo_*.py`: Performs inference with trained models

---

## 💡 Models

- All scripts use: `Qwen2.5-0.5B_v2`
- Fine-tuned checkpoint: `checkpoint-116000`
- Uses Hugging Face `transformers`, `datasets`, `evaluate`

---

## 💾 Notes

- Large `.json` files are tracked via Git LFS
- Ensure `git lfs install` is run before cloning repo
