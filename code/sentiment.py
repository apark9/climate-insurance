import os
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool
from textstat import textstat
import pandas as pd
import logging
import spacy

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(
    filename="sentiment_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

transcript_folder = "input/transcripts"
news_folder = "output/news"
processed_folder = "output/processed"
os.makedirs(processed_folder, exist_ok=True)

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

def analyze_transcripts():
    input_files = [os.path.join(transcript_folder, f) for f in os.listdir(transcript_folder) if f.endswith(".pdf")]
    if not input_files:
        logging.warning("No transcript (PDF) files found for processing.")
        return

    logging.info(f"Found {len(input_files)} transcript files to process.")
    all_results = []

    def analyze_pdf(file_path):
        try:
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() for page in reader.pages])
            filtered_sentences = filter_sentences(text)
            results = []
            for sentence in filtered_sentences:
                sentiment = analyzer.polarity_scores(sentence)
                complexity = calculate_complexity_metrics(sentence)
                results.append({
                    "file": os.path.basename(file_path),
                    "sentence": sentence,
                    "sentiment": sentiment,
                    "flesch_reading_ease": complexity["flesch_reading_ease"],
                    "gunning_fog_index": complexity["gunning_fog_index"],
                    "smog_index": complexity["smog_index"],
                    "lexical_density": complexity["lexical_density"],
                })
            return results
        except Exception as e:
            logging.error(f"Error processing transcript file {file_path}: {e}")
            return []

    with Pool(processes=8) as pool:
        file_results = pool.map(analyze_pdf, input_files)
        for result in file_results:
            all_results.extend(result)

    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_file = os.path.join(processed_folder, "transcript_analysis_results.csv")
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Transcript analysis results saved to {output_file}")
    else:
        logging.warning("No transcript results to save after processing.")

def analyze_news():
    input_files = [os.path.join(news_folder, f) for f in os.listdir(news_folder) if f.endswith(".csv")]
    if not input_files:
        logging.warning("No news (CSV) files found for processing.")
        return

    logging.info(f"Found {len(input_files)} news files to process.")
    all_results = []

    def analyze_csv(file_path):
        try:
            df = pd.read_csv(file_path)
            results = []
            for _, row in df.iterrows():
                snippet = row.get("snippet", "")
                sentiment = analyzer.polarity_scores(snippet)
                complexity = calculate_complexity_metrics(snippet)
                results.append({
                    "file": os.path.basename(file_path),
                    "sentence": snippet,
                    "sentiment": sentiment,
                    "flesch_reading_ease": complexity["flesch_reading_ease"],
                    "gunning_fog_index": complexity["gunning_fog_index"],
                    "smog_index": complexity["smog_index"],
                    "lexical_density": complexity["lexical_density"],
                })
            return results
        except Exception as e:
            logging.error(f"Error processing news file {file_path}: {e}")
            return []

    with Pool(processes=8) as pool:
        file_results = pool.map(analyze_csv, input_files)
        for result in file_results:
            all_results.extend(result)

    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_file = os.path.join(processed_folder, "news_analysis_results.csv")
        combined_df.to_csv(output_file, index=False)
        logging.info(f"News analysis results saved to {output_file}")
    else:
        logging.warning("No news results to save after processing.")

if __name__ == "__main__":
    analyze_news()
    # analyze_transcripts()
