import os
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import logging

# Folders and configurations
interview_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Insurance_Interviews/"
transcript_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Public_Insurance_Transcripts/"
output_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/output"
keywords = ["climate risk", "catastrophe losses", "weather risk", "premium adjustment",
            "reinsurance program", "flood", "hurricane", "global warming"]
analyzer = SentimentIntensityAnalyzer()

def filter_2_sentences(text):
    """
    Extract consecutive sentences where at least one contains a keyword.
    """
    sentences = text.split('.')
    relevant_sentence_pairs = []
    for i in range(len(sentences) - 1):
        sentence1 = sentences[i].strip()
        sentence2 = sentences[i + 1].strip()
        if any(keyword in sentence1.lower() for keyword in keywords) or any(keyword in sentence2.lower() for keyword in keywords):
            relevant_sentence_pairs.append(f"{sentence1}. {sentence2}.")
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

def analyze_pdfs(folder_path, files, result_storage, file_type, ticker=None, quarter=None, year=None):
    for i, file in enumerate(files, start=1):
        # Ensure file is a string before processing
        if not isinstance(file, str) or not file.endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file)

        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
        filtered_sentences = filter_2_sentences(text)

        for sentence in filtered_sentences:
            sentiment = analyzer.polarity_scores(sentence)
            result_storage.append({
                "ticker": ticker,
                "file": file,
                "sentence": sentence,
                "sentiment": sentiment,
                "type": file_type,
                "quarter": quarter,
                "year": year
            })

def process_sentiment():
    interview_results = []
    transcript_results = []

    # Process interviews
    # logging.info("Starting interview analysis...")
    # analyze_pdfs(interview_folder, os.listdir(interview_folder), interview_results, file_type="interview")
    # logging.info("Interview analysis complete.")

    # Process transcripts by ticker
    logging.info("Starting transcript analysis...")
    ticker_files = {}

    # Group files by ticker
    for file in os.listdir(transcript_folder):
        if file.endswith(".pdf"):
            ticker, quarter, year = earnings_calls_split(file)
            if ticker not in ticker_files:
                ticker_files[ticker] = []
            ticker_files[ticker].append((file, quarter, year))  # Include quarter and year

    # Process each ticker
    for ticker, files in ticker_files.items():
        logging.info(f"Processing all files for ticker: {ticker}")
        for file, quarter, year in files:
            analyze_pdfs(transcript_folder, [file], transcript_results, "earnings call", ticker, quarter, year)

    logging.info("Transcript analysis complete. All tickers processed.")

    # Save results to the `output_folder`
    if interview_results:
        interview_output_path = os.path.join(output_folder, "insurance_interviews.csv")
        pd.DataFrame(interview_results).to_csv(interview_output_path, index=False)
        logging.info(f"Interview analysis saved to '{interview_output_path}'.")
    else:
        logging.warning("No interview results to save.")

    if transcript_results:
        transcript_output_path = os.path.join(output_folder, "insurance_transcripts.csv")
        pd.DataFrame(transcript_results).to_csv(transcript_output_path, index=False)
        logging.info(f"Transcript analysis saved to '{transcript_output_path}'.")
    else:
        logging.warning("No transcript results to save.")

if __name__ == "__main__":
    process_sentiment()
