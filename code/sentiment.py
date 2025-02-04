import os
import json
import logging
import pandas as pd
import spacy
import argparse
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool
from textstat import textstat

# Set up logging
os.makedirs("output", exist_ok=True)
logging.basicConfig(
    filename="output/sentiment_analysis.log",
    filemode="w",  # Overwrite on each run
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    force=True,
)

# Load NLP model
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Paths
TRANSCRIPT_FOLDER = "data/transcripts"
OUTPUT_FOLDER = "output"
KEYWORD_FILE = os.path.join(OUTPUT_FOLDER, "custom_keywords.json")  # Load externally defined keywords
CHECKPOINT_FILE = os.path.join(OUTPUT_FOLDER, "sentiment_checkpoint.json")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load keywords from file (or empty list if not defined yet)
if os.path.exists(KEYWORD_FILE):
    with open(KEYWORD_FILE, "r") as f:
        CUSTOM_KEYWORDS = json.load(f)
else:
    CUSTOM_KEYWORDS = []
    logging.warning("No keyword file found. Sentiment analysis will proceed without keyword filtering.")

# Set default context window
DEFAULT_CONTEXT_SENTENCES = 2


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""


def parse_filename(file_name):
    """Extracts stock ticker and quarter/year from filename."""
    try:
        parts = file_name.split("_")
        ticker = parts[0]
        quarter_year = parts[1].replace(".pdf", "")
        quarter = quarter_year[:2]
        year = int("20" + quarter_year[2:])
        return ticker, quarter, year
    except Exception as e:
        logging.error(f"Filename parsing error {file_name}: {e}")
        return None, None, None


def filter_sentences(text, context_sentences=DEFAULT_CONTEXT_SENTENCES):
    """Filters sentences based on keywords and removes those mentioning people."""
    doc = list(nlp.pipe(text.split(". ")))  # Split and process sentences

    filtered_sentences = []
    for i in range(len(doc) - context_sentences + 1):
        sentence_group = " ".join([doc[i + j].text.strip() for j in range(context_sentences)])
        
        # Remove if a person's name is detected
        if any(ent.label_ == "PERSON" for ent in doc[i].ents):
            continue

        # Check if it contains keywords
        if any(kw in sentence_group.lower() for kw in CUSTOM_KEYWORDS):
            filtered_sentences.append(sentence_group)

    return filtered_sentences


def calculate_complexity_metrics(sentence):
    """Calculates readability metrics."""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(sentence),
        "gunning_fog_index": textstat.gunning_fog(sentence),
        "smog_index": textstat.smog_index(sentence),
        "lexical_density": len(set(sentence.split())) / len(sentence.split()) if len(sentence.split()) > 0 else 0,
    }


def analyze_pdf(file_path, context_sentences=DEFAULT_CONTEXT_SENTENCES):
    """Performs sentiment and complexity analysis on a transcript."""
    file_name = os.path.basename(file_path)
    ticker, quarter, year = parse_filename(file_name)
    if not ticker:
        return []

    logging.info(f"Processing {ticker} for {quarter} {year}...")

    text = extract_text_from_pdf(file_path)
    filtered_sentences = filter_sentences(text, context_sentences)

    results = []
    for sentence in filtered_sentences:
        sentiment = analyzer.polarity_scores(sentence)
        complexity = calculate_complexity_metrics(sentence)
        results.append({
            "ticker": ticker,
            "quarter": quarter,
            "year": year,
            "sentence": sentence,
            "sentiment": sentiment["compound"],
            "flesch_reading_ease": complexity["flesch_reading_ease"],
            "gunning_fog_index": complexity["gunning_fog_index"],
            "smog_index": complexity["smog_index"],
            "lexical_density": complexity["lexical_density"],
        })

    return results


def analyze_transcripts(context_sentences=DEFAULT_CONTEXT_SENTENCES):
    """Main function for sentiment analysis."""
    input_files = [os.path.join(TRANSCRIPT_FOLDER, f) for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".pdf")]

    if not input_files:
        logging.warning("No transcripts found.")
        return

    logging.info(f"Found {len(input_files)} transcripts.")

    all_results = []
    with Pool(processes=8) as pool:
        file_results = pool.starmap(analyze_pdf, [(f, context_sentences) for f in input_files])

    for result in file_results:
        all_results.extend(result)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = os.path.join(OUTPUT_FOLDER, "sentiment_results.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"Saved sentiment results to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run sentiment analysis on earnings call transcripts.")
    parser.add_argument("--context_sentences", type=int, default=DEFAULT_CONTEXT_SENTENCES, 
                        help="Number of sentences to use as context (default: 2)")
    args = parser.parse_args()

    analyze_transcripts(context_sentences=args.context_sentences)
