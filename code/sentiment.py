import os
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool
from textstat import textstat
import pandas as pd
import logging
import spacy
from config import keywords, transcript_folder, output_folder, keyword_flag

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

def filter_sentences(text):
    sentences = text.split('.')
    relevant_sentence_pairs = []
    docs = list(nlp.pipe([" ".join(sentences[i:i + 2]).strip() for i in range(len(sentences) - 1)]))

    for i, doc in enumerate(docs):
        sentence_pair = " ".join(sentences[i:i + 2]).strip()
        if any(ent.label_ == "PERSON" for ent in doc.ents):
            continue
        if any(keyword in sentence.lower() for sentence in sentences[i:i + 2] for keyword in keywords):
            relevant_sentence_pairs.append(sentence_pair)

    return relevant_sentence_pairs

def earnings_calls_split(file_name):
    parts = file_name.split('_')
    ticker = parts[0]
    quarter_year = parts[1].replace('.pdf', '')
    quarter = quarter_year[:2]
    year = quarter_year[2:]
    return ticker, quarter, int("20" + year)

def calculate_complexity_metrics(sentence):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(sentence),
        "gunning_fog_index": textstat.gunning_fog(sentence),
        "smog_index": textstat.smog_index(sentence),
        "lexical_density": len(set(sentence.split())) / len(sentence.split()) if len(sentence.split()) > 0 else 0
    }

def analyze_pdf(file_path, file_type, ticker=None, quarter=None, year=None):
    try:
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
        filtered_sentences = filter_sentences(text)
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
                "type": file_type,
                "quarter": quarter,
                "year": year,
            })
        return results

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []

def process_sentiment():
    logging.info("Starting transcript analysis...")
    ticker_files = {}

    for file in os.listdir(transcript_folder):
        if file.endswith(".pdf"):
            ticker, quarter, year = earnings_calls_split(file)
            if ticker not in ticker_files:
                ticker_files[ticker] = []
            ticker_files[ticker].append((os.path.join(transcript_folder, file), "earnings call", ticker, quarter, year))

    tasks = [
        (file_path, file_type, ticker, quarter, year)
        for ticker, files in ticker_files.items()
        for file_path, file_type, ticker, quarter, year in files
    ]

    transcript_results = []
    with Pool(processes=8) as pool:
        all_results = pool.starmap(analyze_pdf, tasks)
        for result in all_results:
            transcript_results.extend(result)

    logging.info("Transcript analysis complete. All tickers processed.")

    if transcript_results:
        complexity_data = pd.DataFrame(transcript_results)
        complexity_data["compound_sentiment"] = complexity_data["sentiment"].apply(lambda x: x["compound"])
        complexity_data.fillna({
            "flesch_reading_ease": 0,
            "gunning_fog_index": 0,
            "smog_index": 0,
            "lexical_density": 0
        }, inplace=True)

        complexity_output_path = os.path.join(output_folder, f"complexity_sentiment_data_{keyword_flag}.csv")
        logging.info(f"Saving complexity data to {complexity_output_path}")
        complexity_data.to_csv(complexity_output_path, index=False)
        logging.info(f"Complexity data saved to {complexity_output_path}")

        aggregated_sentiment = complexity_data.groupby(["ticker", "year"]).agg({
            "compound_sentiment": "mean",
            "flesch_reading_ease": "mean",
            "gunning_fog_index": "mean",
            "smog_index": "mean",
            "lexical_density": "mean"
        }).reset_index().rename(columns={"year": "Year"})

        aggregated_output_path = os.path.join(output_folder, f"aggregated_sentiment_data_{keyword_flag}.csv")
        aggregated_sentiment.to_csv(aggregated_output_path, index=False)
        logging.info(f"Aggregated sentiment and complexity data saved to '{aggregated_output_path}'.")
    else:
        logging.warning("No transcript results to save.")

if __name__ == "__main__":
    process_sentiment()
