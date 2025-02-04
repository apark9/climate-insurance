import os
import json
import pandas as pd
import spacy
from collections import Counter, defaultdict
from PyPDF2 import PdfReader
import logging
from multiprocessing import Pool

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/keyword_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    force=True,
)

# Load SpaCy model with increased max_length
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 4_000_000  # Increase limit for long texts

# Paths
TRANSCRIPT_FOLDER = "data/transcripts"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output files
COMPANY_KEYWORDS_FILE = os.path.join(OUTPUT_FOLDER, "company_keywords.json")
GLOBAL_KEYWORDS_FILE = os.path.join(OUTPUT_FOLDER, "global_keywords.json")
KEYWORD_CSV_FILE = os.path.join(OUTPUT_FOLDER, "keyword_frequencies.csv")

# Define chunk size for splitting large transcripts
CHUNK_SIZE = 500_000  # Process in 500,000 character chunks


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""


def parse_filename(file_name):
    """Extracts ticker from filename (assuming format: TICKER_QuarterYear.pdf)."""
    try:
        parts = file_name.split("_")
        return parts[0]  # Extract stock ticker (company name)
    except Exception as e:
        logging.error(f"Filename parsing error {file_name}: {e}")
        return "UNKNOWN"


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE):
    """Splits large text into smaller chunks for faster processing."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_most_common_words(text, top_n=50):
    """Extracts the most common words from text using batch processing."""
    if not text:
        return []

    try:
        docs = list(nlp.pipe([text], batch_size=5, disable=["parser", "ner"]))  # Faster processing
        words = [token.text.lower() for token in docs[0] if token.is_alpha and not token.is_stop]
        return Counter(words).most_common(top_n)
    except Exception as e:
        logging.error(f"Error in keyword extraction: {e}")
        return []


def process_file(file_path):
    """Processes a single transcript file and extracts keywords."""
    file_name = os.path.basename(file_path)
    company = parse_filename(file_name)  # Get company ticker
    logging.info(f"Processing file: {file_name} (Company: {company})")

    text = extract_text_from_pdf(file_path)
    if not text:
        logging.warning(f"No extractable text found in {file_name}. Skipping...")
        return company, file_name, []

    if len(text) > nlp.max_length:
        logging.warning(f"Text for {file_name} is too long ({len(text)} chars), splitting into chunks.")
        text_chunks = split_text_into_chunks(text)
    else:
        text_chunks = [text]

    keywords = []
    for chunk in text_chunks:
        keywords.extend(get_most_common_words(chunk, top_n=50))

    # Aggregate counts across chunks
    aggregated_keywords = Counter(dict(keywords)).most_common(50)
    return company, file_name, aggregated_keywords


def analyze_keywords():
    """Extracts the most common words per transcript and aggregates them per company and globally."""
    logging.info("Starting keyword analysis...")

    input_files = [os.path.join(TRANSCRIPT_FOLDER, f) for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".pdf")]

    if not input_files:
        logging.warning("No transcripts found for keyword analysis.")
        return

    logging.info(f"Found {len(input_files)} transcripts for keyword analysis.")

    per_file_keywords = {}
    per_company_keywords = defaultdict(Counter)  # Aggregate per company
    global_keywords_counter = Counter()

    # Use multiprocessing to process files in parallel
    with Pool(processes=8) as pool:
        results = pool.map(process_file, input_files)

    for company, file_name, keywords in results:
        if keywords:
            per_file_keywords[file_name] = keywords
            per_company_keywords[company].update(dict(keywords))
            global_keywords_counter.update(dict(keywords))

    # Convert per-company counters to sorted lists
    per_company_keywords = {
        company: counter.most_common(50) for company, counter in per_company_keywords.items()
    }
    global_keywords = global_keywords_counter.most_common(50)

    # Save to JSON
    with open(COMPANY_KEYWORDS_FILE, "w") as f:
        json.dump(per_company_keywords, f, indent=4)

    with open(GLOBAL_KEYWORDS_FILE, "w") as f:
        json.dump(global_keywords, f, indent=4)

    # Save to CSV
    keyword_list = []
    for company, words in per_company_keywords.items():
        for word, count in words:
            keyword_list.append({"company": company, "word": word, "count": count})

    global_keyword_list = [{"company": "ALL", "word": word, "count": count} for word, count in global_keywords]
    keyword_list.extend(global_keyword_list)

    keyword_df = pd.DataFrame(keyword_list)
    keyword_df.to_csv(KEYWORD_CSV_FILE, index=False)

    logging.info(f"Keyword analysis completed. Results saved.")
    logging.info(f"Per-company keywords saved to {COMPANY_KEYWORDS_FILE}")
    logging.info(f"Global keywords saved to {GLOBAL_KEYWORDS_FILE}")
    logging.info(f"Keyword frequencies saved to {KEYWORD_CSV_FILE}")


if __name__ == "__main__":
    analyze_keywords()