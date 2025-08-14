# Task 4 – Named Entity Recognition (NER) from News Articles

## Description
This project implements **Named Entity Recognition (NER)** on news articles using both **rule-based** and **model-based** approaches. The goal is to identify entities such as **people, organizations, locations, dates, emails, and URLs**, and visualize them.

**Bonus tasks completed:**
- ✅ Visualizations with **displaCy** for sample documents
- ✅ Comparison of two spaCy models (`en_core_web_sm` vs `en_core_web_lg`) using Jaccard similarity and label counts

**Recommended dataset:** [CoNLL003 (Kaggle)](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)

---

## 🔧 Requirements

Install Python dependencies:

```bash
pip install pandas spacy tqdm matplotlib
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```
---
## ⚡ How to Run

1. **Edit the CONFIG block in `ner_news_pipeline.py`:**

```python
DATA_DIR = r"./sample_data"
OUTPUT_DIR = r"./ner_outputs"
SAMPLES_TO_VISUALIZE = 15
MAX_DOCS_PER_SPLIT = None  # or a smaller number for testing
```
2. **Run the script:**

```bash
python ner_news_pipeline.py
```
3. **Outputs**

Outputs will be saved in the `ner_outputs/` folder.

---

## 📊 Outputs Explained

| File / Folder                     | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `ner_results_all_splits.csv`      | Master CSV with text previews, entities from rule-based, small model, large model, and Jaccard similarity. |
| `*_rule_label_counts.csv`         | Entity counts per label from rule-based NER.                                |
| `*_sm_label_counts.csv`           | Entity counts per label from `en_core_web_sm`.                               |
| `*_lg_label_counts.csv`           | Entity counts per label from `en_core_web_lg`.                               |
| `model_comparison_summary.csv`    | Comparison of top 5 labels, unique labels per model, and number of documents per split. |
| `sm_vs_lg_jaccard_by_split.csv`   | Jaccard similarity between small and large models per split.                |
| `displacy_html/`                  | HTML visualizations highlighting entities per model/sample.                 |

---

## 🧩 Example Usage

**Load master results in Python:**

```python
import pandas as pd

df = pd.read_csv("./ner_outputs/ner_results_all_splits.csv")
print(df.head())
```
**Visualize an HTML file:**

Open any file in `ner_outputs/displacy_html/` with your browser:

---

## 📈 Visualizations

Here are some sample visualizations:

**Small model (`en_core_web_sm`)**

![Small model visualization](./images/small_model_sample.png)

**Large model (`en_core_web_lg`)**

![Large model visualization](./images/large_model_sample.png)

> Tip: Open the `.html` files in a browser to interactively explore highlighted entities.

---

## ✅ Task Completion

**Implemented features:**

- Load multiple file formats (CSV, TSV, JSONL, CoNLL)  
- Rule-based NER with gazetteers + regex  
- Model-based NER with two spaCy models  
- Save unified CSV of entities  
- Save per-split entity frequency CSVs  
- Generate displaCy HTML visualizations  
- Compare two spaCy models (counts + Jaccard)  
- Bonus: sample visualizations included  

---

## 📖 References

- [spaCy Documentation](https://spacy.io/usage)  
- [displaCy Visualizer](https://spacy.io/usage/visualizers)  
- [CoNLL-2003 Dataset](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)  

---

## ⚠️ Notes

- For large datasets, consider setting `MAX_DOCS_PER_SPLIT` to limit runtime.  
- The rule-based NER uses a small static gazetteer; you can extend it with more entities.  
- HTML visualizations are generated for the first `SAMPLES_TO_VISUALIZE` documents per split.  




























