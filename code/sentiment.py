import os
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool
from textstat import textstat
from config import keyword_flag
import pandas as pd
import logging
import spacy

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

transcript_folder = "data/transcripts"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

def filter_sentences(text):
    sentences = text.split(".")
    relevant_sentence_pairs = []
    docs = list(nlp.pipe([" ".join(sentences[i:i + 2]).strip() for i in range(len(sentences) - 1)]))
    for i, doc in enumerate(docs):
        sentence_pair = " ".join(sentences[i:i + 2]).strip()
        if any(ent.label_ == "PERSON" for ent in doc.ents):
            continue
        if any(keyword in sentence.lower() for sentence in sentences[i:i + 2] for keyword in ["insurance", "risk", "reinsurance", "climate"]):
            relevant_sentence_pairs.append(sentence_pair)
    return relevant_sentence_pairs

def calculate_complexity_metrics(sentence):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(sentence),
        "gunning_fog_index": textstat.gunning_fog(sentence),
        "smog_index": textstat.smog_index(sentence),
        "lexical_density": len(set(sentence.split())) / len(sentence.split()) if len(sentence.split()) > 0 else 0
    }

def parse_filename(file_name):
        parts = file_name.split("_")
        ticker = parts[0]
        quarter_year = parts[1].replace(".pdf", "")
        quarter = quarter_year[:2]
        year = quarter_year[2:]
        return ticker, quarter, int("20" + year)

def analyze_pdf(file_path):
    try:
        logging.info(f"Processing transcript file: {file_path}")
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
        filtered_sentences = filter_sentences(text)
        ticker, quarter, year = parse_filename(os.path.basename(file_path))
        results = []
        for sentence in filtered_sentences:
            sentiment = analyzer.polarity_scores(sentence)
            complexity = calculate_complexity_metrics(sentence)
            results.append({
                "ticker": ticker,
                "file": os.path.basename(file_path),
                "sentence": sentence,
                "sentiment": sentiment,
                "flesch_reading_ease": complexity["flesch_reading_ease"],
                "gunning_fog_index": complexity["gunning_fog_index"],
                "smog_index": complexity["smog_index"],
                "lexical_density": complexity["lexical_density"],
                "type": "earnings call",
                "quarter": quarter,
                "Year": year,
            })
        return results
    except Exception as e:
        logging.error(f"Error processing transcript file {file_path}: {e}")
        return []

def analyze_transcripts():
    input_files = [os.path.join(transcript_folder, f) for f in os.listdir(transcript_folder) if f.endswith(".pdf")]
    if not input_files:
        logging.warning("No transcript (PDF) files found for processing.")
        return

    logging.info(f"Found {len(input_files)} transcript files to process.")
    all_results = []

    with Pool(processes=8) as pool:
        file_results = pool.map(analyze_pdf, input_files)
        for result in file_results:
            all_results.extend(result)

    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_file = os.path.join(output_folder, f"transcript_analysis_results_{keyword_flag}.csv")
        combined_df.to_csv(output_file, index=False, escapechar="\\")
        logging.info(f"Transcript analysis results saved to {output_file}")

        aggregated_output = combined_df.groupby(["ticker", "Year"]).agg({
            "flesch_reading_ease": "mean",
            "gunning_fog_index": "mean",
            "smog_index": "mean",
            "lexical_density": "mean",
            "sentiment": lambda x: pd.Series([s["compound"] for s in x]).mean(),
        }).reset_index()

        aggregated_output_path = os.path.join(output_folder, f"aggregated_transcript_analysis_results_{keyword_flag}.csv")
        aggregated_output.to_csv(aggregated_output_path, index=False)
        logging.info(f"Aggregated transcript analysis results saved to {aggregated_output_path}")
    else:
        logging.warning("No transcript results to save after processing.")

if __name__ == "__main__":
    analyze_transcripts()
