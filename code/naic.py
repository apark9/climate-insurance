import os
import pandas as pd
import logging
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from collections import Counter

# Load NLP Model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Increase max length limit

# Paths
data_folder = "data/NAIC_reports"
output_folder = "output"
naic_folder = os.path.join(output_folder, "naic")
company_policies_folder = os.path.join(output_folder, "company_policies")
os.makedirs(naic_folder, exist_ok=True)
os.makedirs(company_policies_folder, exist_ok=True)

# Output files
extracted_text_file = os.path.join(naic_folder, "naic_extracted_text.csv")
topic_modeling_file = os.path.join(naic_folder, "naic_topic_modeling.csv")
company_identification_file = os.path.join(naic_folder, "naic_company_identification.csv")

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

# Identify companies in the extracted text
def identify_company(df_text):
    """Extract company names from text and avoid processing large text at once."""
    
    company_data = []
    for _, row in df_text.iterrows():
        text, year, file_name = row["text"], row["year"], row["file"]

        # Ensure text is a string and handle large text
        text = str(text) if pd.notna(text) else ""
        if len(text) > nlp.max_length:
            text = text[:nlp.max_length]  # Truncate text to fit within SpaCy's limits

        doc = nlp(text)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        guessed_name = Counter(orgs).most_common(1)[0][0] if orgs else "Unknown"
        company_data.append({"year": year, "file": file_name, "company": guessed_name})

    df_companies = pd.DataFrame(company_data)
    df_companies.to_csv(company_identification_file, index=False)

    logging.info(f"✅ Identified company names for {len(df_companies)} documents.")
    return df_companies

# Merge extracted text with company identification
def merge_extracted_text_and_companies():
    """Merge extracted text with company identification before topic modeling."""
    
    df_text = pd.read_csv(extracted_text_file)
    df_companies = pd.read_csv(company_identification_file)

    df_merged = df_text.merge(df_companies, on=["year", "file"], how="left")
    df_merged.to_csv(extracted_text_file, index=False)

    logging.info(f"✅ Merged extracted text with company identification.")
    return df_merged

# Apply topic modeling
def apply_topic_modeling():
    """Apply topic modeling to the merged dataset."""
    
    logging.info(f"🚀 Running topic modeling on final dataset.")
    df_text = pd.read_csv(extracted_text_file)

    vectorizer = CountVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_text["text"].astype(str))  # Ensure text is a string

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    topic_keywords = [{"Topic": f"Topic {idx+1}", "Keywords": [terms[i] for i in topic.argsort()[-15:]]} for idx, topic in enumerate(lda.components_)]

    df_topics = pd.DataFrame(topic_keywords)
    df_topics.to_csv(topic_modeling_file, index=False)

    logging.info(f"✅ Topic modeling completed and saved.")

# Run full pipeline
def analyze_disclosures():
    """Main function to analyze NAIC disclosures."""
    
    logging.info(f"🚀 Processing all extracted text in one go.")

    df_text = load_extracted_text()
    identify_company(df_text)
    merge_extracted_text_and_companies()
    apply_topic_modeling()

if __name__ == "__main__":
    setup_logging()
    analyze_disclosures()
