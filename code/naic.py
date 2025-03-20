import os
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool
from textstat import textstat
from PyPDF2 import PdfReader, PdfWriter
from config import keyword_flag
import pdfplumber
import pandas as pd
import sys
import spacy
import time
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

transcript_folder = "data/transcripts"
output_folder = "output"
mkt_share_folder = "data/NAIC_market_share"
trimmed_folder = "data/NAIC_market_share/NAIC_market_share_trimmed"
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
        return []

def analyze_transcripts():
    input_files = [os.path.join(transcript_folder, f) for f in os.listdir(transcript_folder) if f.endswith(".pdf")]
    if not input_files:
        return

    all_results = []

    with Pool(processes=8) as pool:
        file_results = pool.map(analyze_pdf, input_files)
        for result in file_results:
            all_results.extend(result)

    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_file = os.path.join(output_folder, f"transcript_analysis_results_{keyword_flag}.csv")
        combined_df.to_csv(output_file, index=False, escapechar="\\")

        aggregated_output = combined_df.groupby(["ticker", "Year"]).agg({
            "flesch_reading_ease": "mean",
            "gunning_fog_index": "mean",
            "smog_index": "mean",
            "lexical_density": "mean",
            "sentiment": lambda x: pd.Series([s["compound"] for s in x]).mean(),
        }).reset_index()

        aggregated_output_path = os.path.join(output_folder, f"aggregated_transcript_analysis_results_{keyword_flag}.csv")
        aggregated_output.to_csv(aggregated_output_path, index=False)
    else:
        print("No transcript results to save after processing.")



def trim_pdf(input_pdf):
    start_time = time.time()
    output_pdf = os.path.join(trimmed_folder, os.path.basename(input_pdf))
    with pdfplumber.open(input_pdf) as pdf:
        writer = PdfWriter()
        page_count = 0
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and "01-Fire" in text:
                logging.info(f"Stopping at page {i} for {input_pdf} (found '01-Fire')")
                break
            writer.add_page(PdfReader(input_pdf).pages[i])
            page_count += 1
        with open(output_pdf, "wb") as f:
            writer.write(f)
    elapsed_time = round(time.time() - start_time, 2)
    return output_pdf

BATCH_SIZE = 2

def parse_market_data(text, year):
    lines = text.split("\n")
    national_data = []
    state_data = []
    capture_national = False
    capture_state = False
    current_state = None

    for line in lines:
        if "PROPERTY AND CASUALTY INSURANCE INDUSTRY" in line and "MARKET SHARE REPORT" in line:
            capture_national = True
            capture_state = False
            continue

        match = re.match(r"^([A-Z\s]+)\s+\d+\s+\d+", line)
        if match and len(line.split()) >= 4:
            current_state = match.group(1).strip()
            capture_state = True
            capture_national = False
            continue

        cols = re.split(r"\s{2,}", line.strip())

        if len(cols) >= 5 and cols[-3].replace(",", "").isdigit() and cols[-2].replace(".", "").isdigit():
            company_name = " ".join(cols[:-3])
            premium_written = cols[-3]
            loss_ratio = cols[-1]

            try:
                premium_written = float(premium_written.replace(",", ""))
                loss_ratio = float(loss_ratio)
            except ValueError:
                continue

            if capture_national:
                national_data.append([company_name, year, premium_written, loss_ratio])
            elif capture_state and current_state:
                state_data.append([current_state, company_name, year, premium_written, loss_ratio])

    return national_data, state_data

def pull_market_data(pdf_file):
    start_time = time.time()
    year = os.path.basename(pdf_file).replace(".pdf", "").strip()
    national_results = []
    state_results = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            national_data, state_data = parse_market_data(full_text, year)
            national_results.extend(national_data)
            state_results.extend(state_data)

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

    elapsed_time = round(time.time() - start_time, 2)
    return national_results, state_results

def analyze_single_batch(batch_index):
    trimmed_files = sorted([os.path.join(trimmed_folder, f) for f in os.listdir(trimmed_folder) if f.endswith(".pdf")])

    if not trimmed_files:
        print("No trimmed PDFs found. Exiting.")
        return

    total_batches = (len(trimmed_files) + BATCH_SIZE - 1) // BATCH_SIZE

    if batch_index >= total_batches:
        print(f"Invalid batch index: {batch_index}. Only {total_batches} batch(es) available.")
        return

    batch_start = batch_index * BATCH_SIZE
    batch_files = trimmed_files[batch_start: batch_start + BATCH_SIZE]

    all_national_results = []
    all_state_results = []

    for pdf in batch_files:
        national_data, state_data = pull_market_data(pdf)
        all_national_results.extend(national_data)
        all_state_results.extend(state_data)

    if all_national_results:
        national_df = pd.DataFrame(all_national_results, columns=["Company", "Year", "Premium Written", "Loss Ratio"])
        national_df["Year"] = national_df["Year"].astype(int)

        national_csv = os.path.join(output_folder, f"national_market_share_batch_{batch_index}.csv")
        national_xlsx = os.path.join(output_folder, f"national_market_share_batch_{batch_index}.xlsx")

        national_df.to_csv(national_csv, index=False)
        national_df.to_excel(national_xlsx, index=False)

    if all_state_results:
        state_df = pd.DataFrame(all_state_results, columns=["State", "Company", "Year", "Premium Written", "Loss Ratio"])
        state_df["Year"] = state_df["Year"].astype(int)

        state_csv = os.path.join(output_folder, f"state_market_share_batch_{batch_index}.csv")
        state_xlsx = os.path.join(output_folder, f"state_market_share_batch_{batch_index}.xlsx")

        state_df.to_csv(state_csv, index=False)
        state_df.to_excel(state_xlsx, index=False)

if __name__ == "__main__":
    analyze_transcripts()
    analyze_single_batch(batch_index)