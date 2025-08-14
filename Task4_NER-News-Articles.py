"""
Task 4 – Named Entity Recognition (NER) from News Articles
Bonus: displaCy visualizations + compare two spaCy models

What this script does
---------------------
1) Loads dataset splits (train/valid/test/metadata) from your computer.
   - Accepts CSV/TSV/JSONL with a 'text' column OR CoNLL IOB files (token-per-line, blank-line-separated).
2) Runs:
   - Rule-based NER (simple gazetteers + regexes).
   - Model-based NER with two spaCy models (e.g., en_core_web_sm and en_core_web_lg).
3) Saves:
   - A unified CSV of texts + entities from rule-based, sm, and lg models.
   - Per-split entity frequency CSVs for each model.
   - displaCy HTML visualizations for sample docs (per model).
   - A simple comparison report (entity counts and overlap).

How to run
----------
1) Install deps:
   pip install pandas spacy tqdm matplotlib
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg   # or change MODEL_LARGE below

2) Edit the CONFIG block (paths/models/samples_to_visualize) then run:
   python ner_news_pipeline.py

Notes
-----
- If your files don’t end with typical extensions, set explicit FILE_MAP paths.
- CoNLL parsing expects at least a token per line; if tags are present, they’re ignored for building raw text.
"""

import os
import re
import json
import glob
import pathlib
from typing import List, Tuple, Dict, Iterable
import pandas as pd
from tqdm import tqdm
import spacy
from spacy import displacy
from collections import Counter, defaultdict

try:
    nlp_sm = spacy.load("en_core_web_sm")
except OSError:
    print("Installing en_core_web_sm...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp_sm = spacy.load("en_core_web_sm")

try:
    nlp_md = spacy.load("en_core_web_md")
except OSError:
    print("Installing en_core_web_md...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    nlp_md = spacy.load("en_core_web_md")

# =========================
# ======== CONFIG =========
# =========================
DATA_DIR = r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Tasks\Task4\DataSet" 
OUTPUT_DIR = r"./ner_outputs"               # results & HTML visualizations

# If your files have unusual names or locations, set them explicitly here.
# Otherwise, the script will try to auto-find by common names.
FILE_MAP = {
    "train": None,     
    "valid": None,     
    "test":  None,    
    "metadata": None,  
}

# Two spaCy models to compare
MODEL_SMALL = "en_core_web_sm"
MODEL_LARGE = "en_core_web_lg"  # change to another installed model if you prefer

# Number of samples per split to visualize with displaCy (per model)
SAMPLES_TO_VISUALIZE = 15

# Max docs to process per split (set to None for all)
MAX_DOCS_PER_SPLIT = None  # e.g., 1000

# =========================
# ===== Helper Utils ======
# =========================

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def guess_text_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"text", "article", "content", "body"}]
    if candidates:
        return candidates[0]
    # fallback: largest string-ish column
    str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if str_cols:
        # choose the column with the longest average length
        avg_len = {c: df[c].fillna("").map(len).mean() for c in str_cols}
        return max(avg_len, key=avg_len.get)
    raise ValueError("No suitable text column found. Please rename your text column to 'text' or set FILE_MAP explicitly.")

def load_structured_file(path: str) -> List[str]:
    """
    Load CSV/TSV/JSONL with a 'text' column (or guessed).
    """
    ext = pathlib.Path(path).suffix.lower()
    if ext in [".csv"]:
        df = pd.read_csv(path)
        text_col = guess_text_col(df)
        return df[text_col].dropna().astype(str).tolist()
    elif ext in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t")
        text_col = guess_text_col(df)
        return df[text_col].dropna().astype(str).tolist()
    elif ext in [".jsonl", ".json"]:
        # Try JSONL first
        texts = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # try common keys
                        if "text" in obj:
                            texts.append(str(obj["text"]))
                        else:
                            # fallback: try to combine string-like fields
                            str_vals = [str(v) for v in obj.values() if isinstance(v, str)]
                            if str_vals:
                                texts.append(" ".join(str_vals))
                    elif isinstance(obj, str):
                        texts.append(obj)
        except json.JSONDecodeError:
            # Maybe a full JSON array
            data = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict) and "text" in obj:
                        texts.append(str(obj["text"]))
                    elif isinstance(obj, str):
                        texts.append(obj)
        return [t for t in texts if t and t.strip()]
    else:
        raise ValueError(f"Unsupported structured file type: {path}")

def parse_conll_sentences(path: str) -> List[str]:
    """
    Very simple CoNLL parser:
    - Assumes token-per-line; columns separated by spaces/tabs.
    - Blank line separates sentences.
    - Reconstruct raw text by joining tokens with spaces (basic; not detokenized).
    """
    sentences = []
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                if tokens:
                    sentences.append(" ".join(tokens))
                    tokens = []
                continue
            parts = re.split(r"\s+", ln)
            token = parts[0]
            tokens.append(token)
    if tokens:
        sentences.append(" ".join(tokens))
    return sentences

def load_split_files(data_dir: str, name_hint: str) -> List[str]:
    """
    Try to auto-detect files for a split (train/valid/test/metadata) and load texts.
    Handles CSV/TSV/JSONL or CoNLL-like text files.
    """
    # If explicit file provided, use it
    explicit = FILE_MAP.get(name_hint)
    if explicit:
        return load_any(explicit)

    # Otherwise, search
    patterns = [
        f"*{name_hint}.*",
        f"*{name_hint[:3]}.*" if len(name_hint) >= 3 else f"*{name_hint}.*",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(data_dir, pat)))

    # Prioritize by extension
    priority_ext = [".csv", ".tsv", ".tab", ".jsonl", ".json", ".txt", ".conll", ".iob"]
    candidates = sorted(candidates, key=lambda p: priority_ext.index(pathlib.Path(p).suffix.lower()) if pathlib.Path(p).suffix.lower() in priority_ext else 999)

    for path in candidates:
        try:
            return load_any(path)
        except Exception:
            continue

    # Fallback: try any file with the hint in name
    generic = glob.glob(os.path.join(data_dir, f"*{name_hint}*"))
    for path in generic:
        try:
            return load_any(path)
        except Exception:
            continue

    return []

def load_any(path: str) -> List[str]:
    ext = pathlib.Path(path).suffix.lower()
    if ext in [".csv", ".tsv", ".tab", ".jsonl", ".json"]:
        return load_structured_file(path)
    else:
        # treat as CoNLL/IOB/plain
        # If it's a plain text with one doc per line, collect non-empty lines
        # If it's CoNLL-style, parse into sentences
        # Heuristic: if many blank lines + many short tokens -> CoNLL
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        if re.search(r"\n\s*\n", txt) and re.search(r"^\S+\s+\S+", txt, flags=re.M):
            # looks like token-per-line with blanks
            return parse_conll_sentences(path)
        else:
            # one doc per line
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return lines

# =========================
# Rule-based NER (simple)
# =========================

PEOPLE = {"Biden","Trump","Obama","Putin","Zelenskyy","Elon Musk","Taylor Swift","Messi","Ronaldo"}
ORGS   = {"Google","Apple","Microsoft","Amazon","Meta","OpenAI","UN","EU","NATO","NASA","WHO"}
LOCS   = {"New York","London","Paris","Berlin","Tokyo","Beijing","Cairo","Dubai","Abu Dhabi","Giza","San Francisco"}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE   = re.compile(r"\bhttps?://\S+\b")
MONEY_RE = re.compile(r"\$\s?\d+(?:[.,]\d+)*(?:\s?(?:million|billion|trillion|k|m|bn))?", re.I)
DATE_RE  = re.compile(r"\b(?:\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s\d{1,2})(?:,\s?\d{2,4})?|\b\d{4}-\d{2}-\d{2}\b", re.I)

def rule_based_ner(text: str) -> List[Tuple[str, str]]:
    ents = []
    # Gazetteers (exact or substring match)
    for p in PEOPLE:
        if p in text:
            ents.append((p, "PERSON"))
    for o in ORGS:
        if o in text:
            ents.append((o, "ORG"))
    for l in LOCS:
        if l in text:
            ents.append((l, "GPE"))  # spaCy label for geopolitical entity

    # Regex patterns
    for m in EMAIL_RE.finditer(text):
        ents.append((m.group(0), "EMAIL"))
    for m in URL_RE.finditer(text):
        ents.append((m.group(0), "URL"))
    for m in MONEY_RE.finditer(text):
        ents.append((m.group(0), "MONEY"))
    for m in DATE_RE.finditer(text):
        ents.append((m.group(0), "DATE"))

    # Deduplicate overlapping exact matches
    seen = set()
    dedup = []
    for span, label in ents:
        key = (span, label)
        if key not in seen:
            seen.add(key)
            dedup.append((span, label))
    return dedup

# =========================
# spaCy model helpers
# =========================

def ents_from_doc(doc) -> List[Tuple[str, str, int, int]]:
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

def label_counts(ents: List[Tuple[str, str, int, int]]) -> Counter:
    return Counter([e[1] for e in ents])

def jaccard_on_spans(ents_a, ents_b) -> float:
    """
    Jaccard on character-span tuples (start,end,label).
    """
    set_a = {(s, e, l) for _, l, s, e in [(t, l, sc, ec) for (t, l, sc, ec) in ents_a]}
    set_b = {(s, e, l) for _, l, s, e in [(t, l, sc, ec) for (t, l, sc, ec) in ents_b]}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def stringify_ents(ents: List[Tuple[str, str, int, int]]) -> str:
    return json.dumps([{"text": t, "label": l, "start": s, "end": e} for (t, l, s, e) in ents], ensure_ascii=False)

# =========================
# Main pipeline
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    html_dir = os.path.join(OUTPUT_DIR, "displacy_html")
    ensure_dir(html_dir)

    print("Loading data splits...")
    data_splits = {}
    for split in ["train", "valid", "test", "metadata"]:
        texts = load_split_files(DATA_DIR, split)
        if MAX_DOCS_PER_SPLIT is not None:
            texts = texts[:MAX_DOCS_PER_SPLIT]
        data_splits[split] = texts
        print(f"  {split}: {len(texts)} docs")

    total_docs = sum(len(v) for v in data_splits.values())
    if total_docs == 0:
        raise RuntimeError("No documents loaded. Check DATA_DIR/FILE_MAP or file formats.")

    print("Loading spaCy models...")
    nlp_sm = spacy.load(MODEL_SMALL, disable=[])  # keep defaults
    nlp_lg = spacy.load(MODEL_LARGE, disable=[])

    rows = []
    comparison_rows = []

    for split, texts in data_splits.items():
        print(f"\nProcessing split: {split} ({len(texts)} docs)")
        ent_counter_sm = Counter()
        ent_counter_lg = Counter()
        ent_counter_rule = Counter()

        # Visualization sampling
        vis_indices = list(range(min(SAMPLES_TO_VISUALIZE, len(texts))))

        for i, text in enumerate(tqdm(texts, desc=f"{split} docs")):
            text = str(text)

            # Rule-based
            rb = rule_based_ner(text)
            # convert to consistent tuple with spans (approximate span using search; if multiple, first match)
            rb_spans = []
            for span_text, lbl in rb:
                m = re.search(re.escape(span_text), text)
                if m:
                    rb_spans.append((span_text, lbl, m.start(), m.end()))
                else:
                    rb_spans.append((span_text, lbl, -1, -1))

            # Model-based
            doc_sm = nlp_sm(text)
            ents_sm = ents_from_doc(doc_sm)
            doc_lg = nlp_lg(text)
            ents_lg = ents_from_doc(doc_lg)

            # Counts
            ent_counter_rule.update([l for (_, l, _, _) in rb_spans])
            ent_counter_sm.update([l for (_, l, _, _) in ents_sm])
            ent_counter_lg.update([l for (_, l, _, _) in ents_lg])

            # Simple overlap score between models
            jacc = jaccard_on_spans(ents_sm, ents_lg)

            # Save row (truncate preview text for CSV readability)
            prev = text if len(text) <= 500 else text[:500] + " ..."
            rows.append({
                "split": split,
                "index": i,
                "text_preview": prev,
                "rule_entities": json.dumps([{"text": t, "label": l} for (t, l, _, _) in rb_spans], ensure_ascii=False),
                "sm_entities": stringify_ents(ents_sm),
                "lg_entities": stringify_ents(ents_lg),
                "sm_lg_jaccard": jacc
            })

            # Visualization: save HTML for first N docs
            if i in vis_indices:
                # Small model
                html_sm = displacy.render(doc_sm, style="ent", page=True)
                with open(os.path.join(html_dir, f"{split}_sm_{i}.html"), "w", encoding="utf-8") as f:
                    f.write(html_sm)
                # Large model
                html_lg = displacy.render(doc_lg, style="ent", page=True)
                with open(os.path.join(html_dir, f"{split}_lg_{i}.html"), "w", encoding="utf-8") as f:
                    f.write(html_lg)

        # Save per-split counts
        pd.DataFrame.from_dict(ent_counter_rule, orient="index", columns=["count"]).sort_values("count", ascending=False)\
            .to_csv(os.path.join(OUTPUT_DIR, f"{split}_rule_label_counts.csv"))
        pd.DataFrame.from_dict(ent_counter_sm, orient="index", columns=["count"]).sort_values("count", ascending=False)\
            .to_csv(os.path.join(OUTPUT_DIR, f"{split}_sm_label_counts.csv"))
        pd.DataFrame.from_dict(ent_counter_lg, orient="index", columns=["count"], ) .sort_values("count", ascending=False)\
            .to_csv(os.path.join(OUTPUT_DIR, f"{split}_lg_label_counts.csv"))

        # Comparison summary row
        comparison_rows.append({
            "split": split,
            "num_docs": len(texts),
            "rule_unique_labels": len(ent_counter_rule),
            "sm_unique_labels": len(ent_counter_sm),
            "lg_unique_labels": len(ent_counter_lg),
            "top5_rule": ", ".join([f"{lbl}:{cnt}" for lbl, cnt in ent_counter_rule.most_common(5)]),
            "top5_sm":   ", ".join([f"{lbl}:{cnt}" for lbl, cnt in ent_counter_sm.most_common(5)]),
            "top5_lg":   ", ".join([f"{lbl}:{cnt}" for lbl, cnt in ent_counter_lg.most_common(5)]),
        })

    # Save master CSV
    master_df = pd.DataFrame(rows)
    master_path = os.path.join(OUTPUT_DIR, "ner_results_all_splits.csv")
    master_df.to_csv(master_path, index=False)

    # Save comparison summary
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison_summary.csv")
    comp_df.to_csv(comp_path, index=False)

    # Also compute overall Jaccard stats per split
    jacc_stats = master_df.groupby("split")["sm_lg_jaccard"].agg(["count", "mean", "median"])
    jacc_stats.to_csv(os.path.join(OUTPUT_DIR, "sm_vs_lg_jaccard_by_split.csv"))

    print("\nDone.")
    print(f"- Master results: {master_path}")
    print(f"- Split label counts: *_rule_label_counts.csv, *_sm_label_counts.csv, *_lg_label_counts.csv")
    print(f"- Comparison summary: {comp_path}")
    print(f"- Jaccard stats: {os.path.join(OUTPUT_DIR, 'sm_vs_lg_jaccard_by_split.csv')}")
    print(f"- displaCy HTML saved under: {html_dir}")
    print("\nTip: open a few of the HTML files to see highlighted entities for each model.")

if __name__ == "__main__":
    main()
