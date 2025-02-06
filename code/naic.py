import os
import pandas as pd
import pytesseract
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
os.makedirs(output_folder, exist_ok=True)

# Output files
extracted_text_file = os.path.join(output_folder, "naic_extracted_text.csv")
topic_modeling_file = os.path.join(output_folder, "naic_topic_modeling.csv")
climate_policies_file = os.path.join(output_folder, "naic_climate_policies.csv")
company_identification_file = os.path.join(output_folder, "naic_company_identification.csv")
climate_policy_trends_file = os.path.join(output_folder, "climate_policy_trends.csv")

# Extract text from a single PDF file
def process_pdf(pdf_info):
    """Processes a single PDF file using OCR."""
    year, pdf_file = pdf_info
    pdf_path = os.path.join(data_folder, year, pdf_file)
    print(f"📄 Processing: {pdf_file}")

    try:
        images = convert_from_path(pdf_path, dpi=150)  # ✅ Lower DPI for less memory use
        extracted_text = "\n".join([pytesseract.image_to_string(img, lang='eng', config="--psm 3 --oem 1") for img in images])
        return {"year": year, "file": pdf_file, "text": extracted_text}
    except Exception as e:
        print(f"❌ ERROR processing {pdf_file}: {e}")
        return {"year": year, "file": pdf_file, "text": ""}

# Process PDFs in batches of 50
def extract_text_from_pdfs(batch_size=50):
    """Processes PDFs in batches and saves progress after each batch."""
    pdf_files = []
    for year in sorted(os.listdir(data_folder)):
        year_path = os.path.join(data_folder, year)
        if not os.path.isdir(year_path):
            continue
        pdf_files.extend([(year, f) for f in os.listdir(year_path) if f.endswith(".pdf")])

    print(f"🚀 Found {len(pdf_files)} PDFs. Processing in batches of {batch_size}...")

    results = []
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i : i + batch_size]
        print(f"📦 Processing batch {i//batch_size + 1} ({len(batch)} PDFs)")

        with Pool(processes=min(6, cpu_count())) as pool:  # ✅ Use max 6 cores
            batch_results = pool.map_async(process_pdf, batch, chunksize=5).get()

        results.extend(batch_results)

        # ✅ Save progress after every batch
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(extracted_text_file, index=False)
        print(f"✅ Saved progress after {i + batch_size} PDFs")

    return df_partial

# Identify the company name from text using Named Entity Recognition (NER)
def identify_company(df_text):
    """Extracts company names from text using NLP."""
    company_data = []

    for _, row in df_text.iterrows():
        text = row["text"]
        year = row["year"]
        file_name = row["file"]

        # Extract potential company names using Named Entity Recognition (NER)
        doc = nlp(text)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        most_common_org = Counter(orgs).most_common(1)

        # If no organization is found, use top frequent words as fallback
        guessed_name = most_common_org[0][0] if most_common_org else "Unknown"

        company_data.append({"year": year, "file": file_name, "company": guessed_name})

    df_companies = pd.DataFrame(company_data)
    df_companies.to_csv(company_identification_file, index=False)
    return df_companies

# Apply topic modeling to detect key climate-related phrases
def apply_topic_modeling(df_text):
    """Finds key themes in extracted text using Topic Modeling (LDA)."""
    vectorizer = CountVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_text["text"])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    topic_keywords = []
    for idx, topic in enumerate(lda.components_):
        top_words = [terms[i] for i in topic.argsort()[-15:]]
        topic_keywords.append({"Topic": f"Topic {idx+1}", "Keywords": top_words})

    df_topics = pd.DataFrame(topic_keywords)
    df_topics.to_csv(topic_modeling_file, index=False)
    return df_topics

# Extract specific climate policies based on topic modeling output
def extract_climate_policies(df_text, df_topics):
    """Extracts key climate policies from text using discovered topic words."""
    topic_keywords = set(df_topics["Keywords"].explode().dropna().unique())

    def extract_statements(text):
        sentences = text.split(".")
        return [sent.strip() for sent in sentences if any(word in sent.lower() for word in topic_keywords)]

    df_text["climate_statements"] = df_text["text"].apply(extract_statements)

    df_text.to_csv(climate_policies_file, index=False)
    return df_text

# Track climate commitments over time
def track_climate_commitments(df_text):
    """Tracks climate policy mentions across years."""
    df_time_series = df_text.groupby("year")["climate_statements"].count().reset_index()
    df_time_series.to_csv(climate_policy_trends_file, index=False)
    return df_time_series

# Full Pipeline Execution
def analyze_disclosures():
    print("Extracting text from PDFs...")
    df_text = extract_text_from_pdfs(batch_size=50)  # ✅ Process in batches of 50

    print("Identifying company names...")
    df_companies = identify_company(df_text)

    print("Applying topic modeling to detect key phrases...")
    df_topics = apply_topic_modeling(df_text)

    print("Extracting climate policies based on detected topics...")
    df_climate_policies = extract_climate_policies(df_text, df_topics)

    print("Tracking climate commitments over time...")
    df_trends = track_climate_commitments(df_climate_policies)

    print("✅ Processing complete. Data saved in the output folder.")

# Run the pipeline
if __name__ == "__main__":
    analyze_disclosures()
