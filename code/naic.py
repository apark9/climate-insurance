import os
import pandas as pd
import logging
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy
from collections import defaultdict

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
policy_sentences_file = os.path.join(naic_folder, "extracted_policy_sentences.csv")
topic_modeling_file = os.path.join(naic_folder, "naic_topic_modeling.csv")

# **Expanded ESG Policy Keywords**
esg_keywords = {
    "climate_risk": ["climate risk", "transition risk", "physical risk", "heat stress", "disaster risk"],
    "sustainable_investment": ["green bond", "impact investing", "sustainable finance", "climate-aligned investment"],
    "carbon_emissions": ["net-zero", "carbon sequestration", "carbon offset", "Scope 1 emissions", "Scope 3 emissions"],
    "renewable_energy": ["solar power", "wind farms", "hydropower", "clean energy transition", "energy efficiency"],
    "climate_governance": ["ESG disclosure", "board oversight", "executive ESG accountability"],
    "biodiversity_nature": ["deforestation", "biodiversity loss", "natural capital", "ecosystem services"],
    "human_rights_labor": ["worker rights", "fair wages", "supply chain ethics", "social responsibility"],
}

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
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
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

# Extract ESG-related policy sentences
def extract_esg_policy_sentences():
    """Extract full policy-related sentences with refined categorization."""
    logging.info(f"🚀 Extracting ESG policy sentences.")

    df_text = pd.read_csv(extracted_text_file)
    df_text["text"] = df_text["text"].astype(str).fillna("")

    extracted_sentences = []
    seen_sentences = set()  # Track unique sentences

    for _, row in df_text.iterrows():
        text = row["text"]
        doc = nlp(text)

        for sent in doc.sents:
            sentence_text = sent.text.lower().strip()

            # **Multi-Category Assignment**
            matched_categories = []
            for category, keywords in esg_keywords.items():
                if any(keyword in sentence_text for keyword in keywords):
                    matched_categories.append(category)

            # **Filter Out Generic or Short Sentences**
            if len(sentence_text.split()) < 5:
                continue
            if sentence_text in seen_sentences:
                continue  # Skip duplicates
            seen_sentences.add(sentence_text)

            # **Store Extracted Sentences**
            if matched_categories:
                extracted_sentences.append({
                    "Company": row["file"],
                    "Categories": ", ".join(matched_categories),
                    "Sentence": sentence_text,
                })

    # Convert to DataFrame
    df_policies = pd.DataFrame(extracted_sentences)
    df_policies.to_csv(policy_sentences_file, index=False)
    logging.info(f"✅ Extracted and saved refined ESG policy sentences.")
    plot_policy_categories(df_policies)

def plot_policy_categories(df_policies):
    """Generate a bar chart showing the frequency of ESG policy categories."""
    category_counts = df_policies["Categories"].str.split(", ").explode().value_counts().reset_index()
    category_counts.columns = ["Category", "Frequency"]

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Frequency", y="Category", data=category_counts, palette="Blues_r")
    plt.xlabel("Frequency")
    plt.ylabel("Policy Category")
    plt.title("Most Common ESG Policy Categories")

    plot_path = os.path.join(plots_folder, "policy_category_distribution.png")
    plt.savefig(plot_path)
    logging.info(f"✅ Saved ESG policy category visualization: {plot_path}")
    plt.close()

# ---- KEEP TOPIC MODELING BUT COMMENTED OUT ----
# def apply_topic_modeling():
#     """Apply LDA and NMF topic modeling to extracted text with refined preprocessing."""
# 
#     logging.info(f"🚀 Running topic modeling on final dataset.")
#     df_text = pd.read_csv(extracted_text_file)
#
#     df_text["text"] = df_text["text"].astype(str).fillna("")
#
#     custom_stopwords = set([
#         "risk", "management", "company", "business", "insurance", "climate", "change", "related",
#         "data", "report", "financial", "investment", "industry", "policy"
#     ])
#
#     vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=3000, ngram_range=(1,3))
#     X = vectorizer.fit_transform(df_text["text"])
#
#     lda = LatentDirichletAllocation(n_components=8, random_state=42)
#     lda.fit(X)
#
#     nmf = NMF(n_components=8, random_state=42)
#     nmf.fit(X)
#
#     terms = vectorizer.get_feature_names_out()
#
#     topic_keywords_lda = [
#         {"Topic": f"Topic {idx+1} (LDA)", "Keywords": [terms[i] for i in topic.argsort()[-12:]]}
#         for idx, topic in enumerate(lda.components_)
#     ]
#
#     topic_keywords_nmf = [
#         {"Topic": f"Topic {idx+1} (NMF)", "Keywords": [terms[i] for i in topic.argsort()[-12:]]}
#         for idx, topic in enumerate(nmf.components_)
#     ]
#
#     df_topics = pd.DataFrame(topic_keywords_lda + topic_keywords_nmf)
#     df_topics.to_csv(topic_modeling_file, index=False)
#
#     logging.info(f"✅ Topic modeling completed and saved.")
#
#     # Visualize topic modeling (commented out)
#     # visualize_topics(lda, terms, title="LDA Topic Modeling", filename="lda_barchart.png")
#     # visualize_topics(nmf, terms, title="NMF Topic Modeling", filename="nmf_barchart.png")

# Run full pipeline
def analyze_disclosures():
    logging.info(f"🚀 Processing all extracted text in one go.")
    load_extracted_text()
    extract_esg_policy_sentences()

if __name__ == "__main__":
    setup_logging()
    analyze_disclosures()
