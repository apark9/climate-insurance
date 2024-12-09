import os
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool
import pandas as pd
import logging
import spacy
from config import keywords, transcript_folder, output_folder, keyword_flag

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

def filter_sentences(text):
    """
    Extract consecutive pairs of sentences where at least one contains a keyword.
    """
    sentences = text.split('.')
    relevant_sentence_pairs = []

    # Process consecutive sentence pairs
    docs = list(nlp.pipe([" ".join(sentences[i:i + 2]).strip() for i in range(len(sentences) - 1)]))

    for i, doc in enumerate(docs):
        sentence_pair = " ".join(sentences[i:i + 2]).strip()

        # Skip pairs mentioning PERSON entities
        if any(ent.label_ == "PERSON" for ent in doc.ents):
            continue

        # Check for keywords
        if any(keyword in sentence.lower() for sentence in sentences[i:i + 2] for keyword in keywords):
            relevant_sentence_pairs.append(sentence_pair)

    return relevant_sentence_pairs


def earnings_calls_split(file_name):
    """
    Split filename to extract ticker, quarter, and year.
    """
    parts = file_name.split('_')
    ticker = parts[0]
    quarter_year = parts[1].replace('.pdf', '')
    quarter = quarter_year[:2]
    year = quarter_year[2:]
    return ticker, quarter, int("20" + year)


def analyze_pdf(file_path, file_type, ticker=None, quarter=None, year=None):
    """
    Analyze a single PDF file for sentiment and relevant sentences.
    """
    try:
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
        filtered_sentences = filter_sentences(text)

        results = []
        for sentence in filtered_sentences:
            sentiment = analyzer.polarity_scores(sentence)
            results.append({
                "ticker": ticker,
                "file": os.path.basename(file_path),
                "sentence": sentence,
                "sentiment": sentiment,
                "type": file_type,
                "quarter": quarter,
                "year": year,
            })
        return results

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []


def process_sentiment():
    """
    Process all transcripts and analyze sentiment in parallel.
    """
    logging.info("Starting transcript analysis...")
    ticker_files = {}

    # Group files by ticker
    for file in os.listdir(transcript_folder):
        if file.endswith(".pdf"):
            ticker, quarter, year = earnings_calls_split(file)
            if ticker not in ticker_files:
                ticker_files[ticker] = []
            ticker_files[ticker].append((os.path.join(transcript_folder, file), "earnings call", ticker, quarter, year))

    # Prepare tasks for parallel processing
    tasks = [
        (file_path, file_type, ticker, quarter, year)
        for ticker, files in ticker_files.items()
        for file_path, file_type, ticker, quarter, year in files
    ]

    # Use multiprocessing Pool for parallel processing
    transcript_results = []
    with Pool(processes=8) as pool:  # Adjust the number of processes to match available CPUs
        all_results = pool.starmap(analyze_pdf, tasks)
        for result in all_results:
            transcript_results.extend(result)

    logging.info("Transcript analysis complete. All tickers processed.")

    # Save raw results to the output folder
    if transcript_results:
        transcript_output_path = os.path.join(output_folder, f"insurance_transcripts_{keyword_flag}.csv")
        pd.DataFrame(transcript_results).to_csv(transcript_output_path, index=False)
        logging.info(f"Transcript analysis saved to '{transcript_output_path}'.")

        # Aggregate sentiment by ticker and year
        aggregated_sentiment = pd.DataFrame(transcript_results)
        aggregated_sentiment["compound_sentiment"] = aggregated_sentiment["sentiment"].apply(lambda x: x["compound"])
        aggregated_sentiment = aggregated_sentiment.groupby(["ticker", "year"]).agg({
            "compound_sentiment": "mean"
        }).reset_index().rename(columns={"year": "Year"})

        # Generate a complete range of years for all tickers
        all_years = pd.DataFrame({
            "Year": range(aggregated_sentiment["Year"].min(), aggregated_sentiment["Year"].max() + 1)
        })
        unique_tickers = aggregated_sentiment["ticker"].unique()

        # Cross join all years with all tickers to ensure a complete grid
        complete_grid = pd.MultiIndex.from_product(
            [unique_tickers, all_years["Year"]],
            names=["ticker", "Year"]
        ).to_frame(index=False)

        # Merge the complete grid with the aggregated sentiment data
        aggregated_sentiment = complete_grid.merge(
            aggregated_sentiment,
            on=["ticker", "Year"],
            how="left"
        )

        # Fill missing sentiment scores with 0 or NaN
        aggregated_sentiment["compound_sentiment"] = aggregated_sentiment["compound_sentiment"].fillna(0)

        # Save aggregated sentiment data
        aggregated_output_path = os.path.join(output_folder, f"aggregated_sentiment_data_{keyword_flag}.csv")
        aggregated_sentiment.to_csv(aggregated_output_path, index=False)
        logging.info(f"Aggregated sentiment data saved to '{aggregated_output_path}'.")

        # Log debugging information
        logging.info(f"Aggregated sentiment data shape: {aggregated_sentiment.shape}")
        missing_years = set(all_years["Year"]) - set(aggregated_sentiment["Year"].unique())
        if missing_years:
            logging.warning(f"Missing years in sentiment data: {missing_years}")
    else:
        logging.warning("No transcript results to save.")

if __name__ == "__main__":
    process_sentiment()
