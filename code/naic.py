import os
import pandas as pd
import logging
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import spacy
from collections import Counter

# Load NLP Model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Increase max length limit

# Paths
data_folder = "data/NAIC_reports"
output_folder = "output"
naic_folder = os.path.join(output_folder, "naic")
plots_folder = "plots"
os.makedirs(naic_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# Output files
extracted_text_file = os.path.join(naic_folder, "naic_extracted_text.csv")
topic_keywords_file = os.path.join(naic_folder, "naic_topic_keywords.csv")

# Set up logging
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/naic_full_run.log"

    if logging.root.handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"📂 Logging started: {log_file}")
    sys.stdout.flush()
    return log_file

# Load extracted text
def load_extracted_text():
    """Load all batch-processed extracted text."""
    
    text_files = [f for f in os.listdir(naic_folder) if f.startswith("naic_extracted_text_")]
    if not text_files:
        logging.error(f"❌ ERROR: No extracted text files found in {naic_folder}.")
        sys.exit(1)

    df_text = pd.concat([pd.read_csv(os.path.join(naic_folder, f)) for f in text_files])
    df_text.to_csv(extracted_text_file, index=False)

    logging.info(f"📂 Loaded and merged {len(df_text)} extracted text records.")
    return df_text

# Remove Named Entities (Company Names, Locations, Dates)
def clean_text_with_ner(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if token.ent_type_ not in ["ORG", "GPE", "DATE"]])

def extract_keywords():
    """Extract high-quality keywords using topic modeling optimized for filtering."""

    logging.info(f"🚀 Extracting keywords from topic modeling.")
    df_text = pd.read_csv(extracted_text_file)

    # Preprocess text
    df_text["text"] = df_text["text"].astype(str).fillna("")
    df_text["text"] = df_text["text"].apply(clean_text_with_ner)  # Remove company names, locations, dates

    # **Refined Stopwords**
    custom_stopwords = set([
        "risk", "management", "company", "business", "insurance", "climate", "change", "related",
        "data", "report", "financial", "investment", "industry", "policy", "emissions", "impact", 
        "opportunities", "members", "consider", "including", "potential", "nationwide", "disclosure"
    ])

    # **Use TF-IDF Instead of CountVectorizer with 1-grams to 3-grams**
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=4000, ngram_range=(1,3))
    X = vectorizer.fit_transform(df_text["text"])

    # **Use 6 Topics Instead of 10 to Reduce Fragmentation**
    lda = LatentDirichletAllocation(n_components=6, random_state=42)
    lda.fit(X)

    nmf = NMF(n_components=6, random_state=42)
    nmf.fit(X)

    # Extract meaningful terms
    terms = vectorizer.get_feature_names_out()

    topic_keywords_lda = [
        {"Topic": f"Topic {idx+1} (LDA)", "Keywords": [terms[i] for i in topic.argsort()[-30:]]}
        for idx, topic in enumerate(lda.components_)
    ]

    topic_keywords_nmf = [
        {"Topic": f"Topic {idx+1} (NMF)", "Keywords": [terms[i] for i in topic.argsort()[-30:]]}
        for idx, topic in enumerate(nmf.components_)
    ]

    df_keywords = pd.DataFrame(topic_keywords_lda + topic_keywords_nmf)
    df_keywords.to_csv(topic_keywords_file, index=False)

    logging.info(f"✅ Keyword extraction completed and saved for filtering.")

    # **Save Top Keywords for Sentiment Analysis Filtering**
    save_filtered_keywords(df_keywords)

def save_filtered_keywords(df_keywords):
    """Save extracted keywords as a plain list for filtering in later analysis."""
    all_keywords = set()
    for _, row in df_keywords.iterrows():
        all_keywords.update(row["Keywords"])

    keywords_list_file = os.path.join(naic_folder, "filtered_keywords.txt")
    with open(keywords_list_file, "w") as f:
        f.write("\n".join(sorted(all_keywords)))

    logging.info(f"✅ Filtered keywords saved to {keywords_list_file} for sentiment analysis.")

# Run full pipeline
def analyze_disclosures():
    """Main function to extract keywords for filtering & sentiment analysis."""
    
    logging.info(f"🚀 Processing all extracted text in one go.")

    df_text = load_extracted_text()
    extract_keywords()

if __name__ == "__main__":
    setup_logging()
    analyze_disclosures()
