############################################################
# goodreads_text_features_final.py
# Description:
#   This script performs full text feature extraction on the Goodreads dataset.
#   It installs dependencies (Transformers, Torch, TextBlob, AWSWrangler),
#   loads Parquet data from S3 (via SageMaker Processing Input),
#   and generates a wide variety of linguistic, readability,
#   sentiment, TF-IDF, topic, zero-shot, and embedding features.
#   The enriched dataset is saved back to S3 in Parquet format.
############################################################

############################################################
# STEP 0: Setup and Environment Bootstrapping
############################################################
"""
Purpose:
  • Prepare SageMaker environment and install required packages dynamically.
  • Ensure that NLTK data (VADER) and Transformers models can be loaded properly.

Details:
  - Packages include AWS SDKs, NLP libraries, and ML frameworks.
  - This ensures reproducibility inside the SageMaker container.
"""

import sys, subprocess, os, pandas as pd, gc

def pip_install(pkgs):
    """Utility: installs Python packages in container runtime."""
    print(f"Installing packages: {pkgs}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs])
    print("All requested pip installs finished!", flush=True)

# Core dependencies
pkgs = [
    "boto3", "s3fs", "textstat", "nltk", "transformers==4.44.2", "torch==2.2.2",
    "pyarrow", "awswrangler", "numpy==1.24.1", "pandas==1.1.3",
    "python-dateutil==2.8.1", "textblob", "scikit-learn"
]
pip_install(pkgs)

# Environment bootstrap verification
import nltk
nltk.download("vader_lexicon")
print("------ ENV BOOTSTRAP SUCCESS ------", flush=True)

############################################################
# STEP 1: Input and File Mount Verification
############################################################
"""
Purpose:
  • Discover and validate input parquet files within /opt/ml/processing/input/
  • Prepare output directory for saving final Parquet outputs
"""

def find_parquet_files(directory):
    parquet_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".parquet") or f.endswith(".parquet.snappy"):
                parquet_files.append(os.path.join(root, f))
    return parquet_files

INPUT_DIR = os.getenv("INPUT_DIR", "/opt/ml/processing/input/features/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/opt/ml/processing/output/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Searching recursively for Parquet files in: {INPUT_DIR}", flush=True)
for root, dirs, files in os.walk(INPUT_DIR):
    print(f"Mounted: {root}\n  Dirs: {dirs}\n  Files: {files}")

parquet_files = find_parquet_files(INPUT_DIR)
if not parquet_files:
    raise FileNotFoundError("No .parquet files found!")
print(f"Found parquet files: {parquet_files}")

############################################################
# STEP 2: Feature Extraction Functions
############################################################
"""
Purpose:
  • Define modular NLP feature functions for readability, sentiment,
    lexical diversity, TF-IDF, topic modeling, and embeddings.
  • Include new derived features (word_count, avg_word_len, etc.)
  • Maintain reproducibility and scalability.
"""

import re, string, traceback, uuid, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from pathlib import Path

# Environment tuning variables
MAX_LEN = int(os.getenv("MAX_LEN", "96"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
ROW_CHUNK = int(os.getenv("ROW_CHUNK", "20000"))
HOST = os.getenv("SM_CURRENT_HOST", "host0")

os.environ.setdefault("HF_HOME", "/opt/ml/processing/hf-cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))

WORD_RE = re.compile(r"\w+")
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF" "\u2700-\u27BF" "]", flags=re.UNICODE
)

REQUIRED_COLUMNS = [
    "book_id", "user_id", "review_id", "rating", "rating_count", "review_text",
    "review_length_raw", "review_char_count", "date_added", "n_votes",
    "title", "publication_year", "author_names"
]

def filter_columns(df):
    """Ensure all required columns exist."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[REQUIRED_COLUMNS]

# ---------- Feature Groups ----------
def formatting_features(df, txt):
    """Sentence/paragraph punctuation features."""
    df["sentence_count"] = txt.str.count(r"[.!?]").astype("int32")
    df["paragraph_count"] = txt.str.count("\n").add(1).astype("int32")
    df["quote_count"] = txt.str.count('"').astype("int32")
    df["exclamation_count"] = txt.str.count("!").astype("int32")
    df["question_count"] = txt.str.count(r"\?").astype("int32")

def length_features(df, txt):
    """Character-based metrics."""
    df["punct_count"] = txt.apply(lambda s: sum(ch in string.punctuation for ch in s)).astype("int32")
    df["caps_count"] = txt.apply(lambda s: sum(ch.isupper() for ch in s)).astype("int32")
    df["emoji_count"] = txt.apply(lambda s: len(EMOJI_RE.findall(s))).astype("int32")

def derived_features(df, txt):
    """Custom additional features to keep from Lecture 05 analysis."""
    df["word_count"] = txt.apply(lambda s: len(WORD_RE.findall(s))).astype("int32")
    df["avg_word_len"] = txt.apply(
        lambda s: sum(len(w) for w in WORD_RE.findall(s)) / max(1, len(WORD_RE.findall(s)))
    ).astype("float32")
    df["has_emoji"] = (df.get("emoji_count", 0) > 0).astype("int8")
    cur_year = pd.Timestamp("now").year
    yr = pd.to_numeric(df["publication_year"], errors="coerce")
    df["book_age"] = (cur_year - yr).clip(lower=0).fillna(0).astype("int16")

def lexical_features(df, txt):
    df["lexical_diversity"] = txt.apply(lambda s: len(set(s.split())) / max(1, len(s.split()))).astype("float32")

def readability_features(df, txt):
    """Approximate Flesch-Kincaid Grade."""
    def fk(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        words = WORD_RE.findall(text)
        if not sentences or not words:
            return 0.0
        syllables = sum(len(re.findall(r"[aeiouyAEIOUY]+", w)) for w in words)
        return float(max(0, 0.39*(len(words)/len(sentences)) + 11.8*(syllables/len(words)) - 15.59))
    df["readability_grade"] = txt.apply(fk).astype("float32")

def sentiment_features(df, txt):
    sia = SentimentIntensityAnalyzer()
    sent = txt.apply(sia.polarity_scores)
    df["vader_pos"] = sent.map(lambda d: d["pos"]).astype("float32")
    df["vader_neg"] = sent.map(lambda d: d["neg"]).astype("float32")
    df["vader_neu"] = sent.map(lambda d: d["neu"]).astype("float32")
    df["blob_subjectivity"] = txt.map(lambda s: TextBlob(s).subjectivity).astype("float32")
    df["blob_polarity"] = txt.map(lambda s: TextBlob(s).polarity).astype("float32")

def tfidf_features(df, txt):
    vec = TfidfVectorizer(max_features=300)
    X = vec.fit_transform(txt.fillna("").tolist())
    df["tfidf_mean"] = np.asarray(X.mean(axis=1)).ravel()
    df["tfidf_max"] = np.asarray(X.max(axis=1).toarray()).ravel()

def topic_features(df, txt):
    vec = TfidfVectorizer(max_features=300)
    nmf = NMF(n_components=5, init="nndsvda", random_state=42)
    X = vec.fit_transform(txt.fillna("").tolist())
    T = nmf.fit_transform(X)
    df["topic1_prob"], df["topic2_prob"] = T[:, 0], T[:, 1]

def zeroshot_features(df, txt):
    """Multi-label zero-shot classification."""
    pipe = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
    labels = ["Contains spoilers", "Recommendation", "Critical", "Summary", "Praise", "Complaint"]
    results = pipe(txt.fillna("").tolist(), labels, multi_label=True)
    df["zeroshot_score"] = [max(r["scores"]) if r["scores"] else 0.0 for r in results]

def embedding_features(df, txt):
    """DistilBERT embeddings for semantic meaning."""
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    mdl = AutoModel.from_pretrained("distilbert-base-uncased").eval().to("cpu")
    embs = []
    docs = txt.tolist()
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        with torch.no_grad():
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            out = mdl(**enc).last_hidden_state[:, 0, :].numpy()
            embs.append(out)
    arr = np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)
    df["distilbert_embedding"] = [row.astype(np.float32).tolist() for row in arr]

############################################################
# STEP 3: Core Pipeline and Output
############################################################
"""
Executes all feature steps sequentially, prunes low-signal columns,
and saves the enriched DataFrame chunk as Parquet.
"""

def process_chunk(df):
    txt = df["review_text"].fillna("")
    print("STEP 1: Formatting & Length Features"); formatting_features(df, txt); length_features(df, txt); derived_features(df, txt)
    print("STEP 2: Lexical Features"); lexical_features(df, txt)
    print("STEP 3: Readability Features"); readability_features(df, txt)
    print("STEP 4: Sentiment Features"); sentiment_features(df, txt)
    print("STEP 5: TFIDF Features"); tfidf_features(df, txt)
    print("STEP 6: Topic Model Features"); topic_features(df, txt)
    print("STEP 7: Zero-shot Classification Features"); zeroshot_features(df, txt)
    print("STEP 8: Embedding Features"); embedding_features(df, txt)

    # Remove unnecessary or redundant columns
    PRUNE_COLS = [
        "quote_count", "exclamation_count", "question_count", "caps_count",
        "tfidf_min", "zeroshot_label", "review_text", "title", "author_names",
        "book_id", "user_id", "review_id", "date_added"
    ]
    df = df[[c for c in df.columns if c not in PRUNE_COLS]]
    print(f"DF shape before write_out: {df.shape}", flush=True)
    return df

def write_out(df, base_name):
    if df.empty:
        print("write_out skipped: DataFrame empty!", flush=True)
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    out = os.path.join(OUTPUT_DIR, f"{base_name}_{HOST}_{uuid.uuid4().hex[:8]}.parquet")
    pq.write_table(table, out, compression="snappy")
    print(f"write_out completed → {out}", flush=True)

############################################################
# STEP 4: Main Entry
############################################################
"""
Main entry for SageMaker job execution.
Handles reading, feature extraction, and S3 output write.
"""

def main():
    try:
        print("Python:", sys.version)
        print("Pandas:", pd.__version__)
        print("NumPy:", np.__version__)

        p = parquet_files[0]
        print(f"Processing parquet: {p}")
        df = pd.read_parquet(p)
        print(f"DF shape at initial read: {df.shape}")

        df = filter_columns(df)
        out_df = process_chunk(df)
        write_out(out_df, base_name="features_enriched_full")

        print("------ PROCESSING AND EMBEDDING SUCCESS ------", flush=True)
        del out_df, df; gc.collect()
    except Exception as e:
        print("FATAL ERROR:", e, flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
