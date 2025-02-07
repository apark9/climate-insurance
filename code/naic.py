import os
import pandas as pd
import pytesseract
import logging
import sys
import time
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from collections import Counter
from multiprocessing import Pool, cpu_count

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

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
climate_policies_file = os.path.join(naic_folder, "naic_climate_policies.csv")
company_identification_file = os.path.join(naic_folder, "naic_company_identification.csv")
climate_policy_trends_file = os.path.join(naic_folder, "climate_policy_trends.csv")

# Set up logging
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{timestamp}.log"

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

# Extract text from a single PDF file
def process_pdf(pdf_info):
    year, pdf_file = pdf_info
    pdf_path = os.path.join(data_folder, year, pdf_file)

    logging.info(f"📄 Processing: {pdf_file}")
    sys.stdout.flush()

    try:
        images = convert_from_path(pdf_path, dpi=100)
        extracted_text = "\n".join([pytesseract.image_to_string(img, lang='eng', config="--psm 6 --oem 1") for img in images])
        return {"year": year, "file": pdf_file, "text": extracted_text}
    except Exception as e:
        logging.error(f"❌ ERROR processing {pdf_file}: {e}")
        sys.stdout.flush()
        return {"year": year, "file": pdf_file, "text": ""}

# Process PDFs in batches
def extract_text_from_pdfs(batch_size=50):
    processed_files = set()

    if os.path.exists(extracted_text_file):
        df_existing = pd.read_csv(extracted_text_file)
        processed_files = set(df_existing["file"])
        logging.info(f"✅ Found {len(processed_files)} already processed PDFs. Skipping them.")

    pdf_files = []
    for year in sorted(os.listdir(data_folder)):
        year_path = os.path.join(data_folder, year)
        if not os.path.isdir(year_path):
            continue
        pdf_files.extend([(year, f) for f in os.listdir(year_path) if f.endswith(".pdf") and f not in processed_files])

    logging.info(f"🚀 Found {len(pdf_files)} unprocessed PDFs.")
    sys.stdout.flush()

    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        logging.info(f"📦 Processing batch {i//batch_size + 1} ({len(batch)} PDFs)")
        sys.stdout.flush()

        with Pool(processes=min(6, cpu_count())) as pool:
            batch_results = pool.map(process_pdf, batch)

        df_new = pd.DataFrame(batch_results)
        df_combined = pd.concat([df_existing, df_new]) if len(processed_files) > 0 else df_new
        df_combined.to_csv(extracted_text_file, index=False)

        logging.info(f"✅ Saved extracted text for {i + batch_size} PDFs.")
        sys.stdout.flush()

    return df_combined

# Identify company names using NLP
def identify_company(df_text):
    company_data = []

    for _, row in df_text.iterrows():
        text, year, file_name = row["text"], row["year"], row["file"]
        doc = nlp(text)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        guessed_name = Counter(orgs).most_common(1)[0][0] if orgs else "Unknown"

        company_data.append({"year": year, "file": file_name, "company": guessed_name})

    df_companies = pd.DataFrame(company_data)
    df_companies.to_csv(company_identification_file, index=False)
    return df_companies

# Apply topic modeling
def apply_topic_modeling():
    if not os.path.exists(extracted_text_file):
        logging.warning("🚨 No extracted text found! Skipping topic modeling.")
        return pd.DataFrame()

    df_text = pd.read_csv(extracted_text_file)

    vectorizer = CountVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_text["text"])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    terms = vectorizer.get_feature_names()
    topic_keywords = [{"Topic": f"Topic {idx+1}", "Keywords": [terms[i] for i in topic.argsort()[-15:]]} for idx, topic in enumerate(lda.components_)]

    df_topics = pd.DataFrame(topic_keywords)
    df_topics.to_csv(topic_modeling_file, index=False)
    return df_topics

# Extract climate policies
def extract_climate_policies(df_text, df_topics):
    topic_keywords = set(df_topics["Keywords"].explode().dropna().unique())

    def extract_statements(text):
        sentences = text.split(".")
        return [sent.strip() for sent in sentences if any(word in sent.lower() for word in topic_keywords)]

    df_text["climate_statements"] = df_text["text"].apply(extract_statements)
    df_text.to_csv(climate_policies_file, index=False)
    return df_text

# Save climate policies per company and year
def save_policies_by_company(df_text, df_companies):
    df_combined = df_text.merge(df_companies, on=["year", "file"], how="left")

    for company, df_group in df_combined.groupby("company"):
        company_cleaned = company.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(company_policies_folder, f"{company_cleaned}_climate_policies.csv")

        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            df_group = pd.concat([df_existing, df_group])

        df_group.to_csv(output_path, index=False)
        logging.info(f"✅ Updated climate policies for {company} ({len(df_group)} records) → {output_path}")

# Track climate commitments
def track_climate_commitments(df_text):
    df_time_series = df_text.groupby("year")["climate_statements"].count().reset_index()
    df_time_series.to_csv(climate_policy_trends_file, index=False)
    return df_time_series

# Full pipeline execution
def analyze_disclosures():
    logging.info("🚀 Starting NAIC disclosures processing...")
    df_text = extract_text_from_pdfs()
    df_companies = identify_company(df_text)
    df_topics = apply_topic_modeling()
    df_climate_policies = extract_climate_policies(df_text, df_topics)
    save_policies_by_company(df_climate_policies, df_companies)
    track_climate_commitments(df_climate_policies)
    logging.info("✅ NAIC disclosures processing complete.")

# Run the pipeline
if __name__ == "__main__":
    setup_logging()
    analyze_disclosures()
