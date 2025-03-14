import os
import logging
import pandas as pd
import spacy
import traceback
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from transformers import pipeline, BertTokenizer
from textstat import textstat
from config import keywords_financial, keywords_climate, keywords_risk, sentiment_flag
from nltk.sentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
finbert = None
vader = SentimentIntensityAnalyzer()

TRANSCRIPT_DATA = f"data/transcripts"
SELL_SIDE_DATA = f"data/sell_side_txt"
OUTPUT_FOLDER = f"data/{sentiment_flag}_output"
ANALYSIS_FOLDER = "analysis/sentiment"
KEYWORD_FOLDER = "analysis/keyword_freq"

NUM_PDFS = 100
MAX_TOKENS = 512
TOTAL_BATCHES = 10

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def load_finbert():
    global finbert
    if finbert is None:
        # finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", return_all_scores=True)
        finbert = pipeline("text-classification", model="ProsusAI/finbert")
        logging.info("FinBERT loaded.")

def extract_text_from_txt(file_path):
    """ Extracts text from a plain text file (used for sell-side reports). """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return ""

def extract_date_from_filename(filename):
    try:
        parts = filename.split("_")
        return parts[0], int("20" + parts[1][2:4]), parts[1][:2]
    except Exception as e:
        logging.error(f"Filename error: {e}")
        return None, None, None

def extract_ticker_year_quarter(filename):
    """ Extracts Ticker, Year, and Quarter from filenames like 'PGR_APR22.txt' and 'AIG_1Q23.txt'. """
    try:
        parts = filename.split("_")
        if len(parts) < 2:
            return None, None, None  # Invalid format

        ticker = parts[0]  # Extract ticker
        identifier = parts[1].split(".")[0].upper()  # Remove file extension and get identifier

        # ✅ Check if format is "MMMYY" (e.g., "APR22")
        if len(identifier) == 5 and identifier[:3].isalpha() and identifier[3:5].isdigit():
            month_str = identifier[:3]  # Extract three-letter month
            year_str = "20" + identifier[3:5]  # Extract last 2 digits of the year
            year = int(year_str)

            # ✅ Convert month to quarter
            month_to_quarter = {
                "JAN": "Q1", "FEB": "Q1", "MAR": "Q1",
                "APR": "Q2", "MAY": "Q2", "JUN": "Q2",
                "JUL": "Q3", "AUG": "Q3", "SEP": "Q3",
                "OCT": "Q4", "NOV": "Q4", "DEC": "Q4"
            }
            quarter = month_to_quarter.get(month_str, None)
            if quarter is None:
                raise ValueError(f"Invalid month format in {filename}: {month_str}")

        # ✅ Check if format is "XQYY" (e.g., "1Q22")
        elif len(identifier) == 4 and identifier[0].isdigit() and identifier[1] == "Q" and identifier[2:].isdigit():
            quarter = f"Q{identifier[0]}"  # Extract quarter
            year_str = "20" + identifier[2:]  # Extract last 2 digits of the year
            year = int(year_str)

        else:
            raise ValueError(f"Unknown format in {filename}: {identifier}")

        return ticker, year, quarter

    except Exception as e:
        logging.error(f"Filename error in {filename}: {e}")
        return None, None, None

def calculate_readability(sentence):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(sentence),
        "gunning_fog_index": textstat.gunning_fog(sentence),
        "smog_index": textstat.smog_index(sentence),
        "lexical_density": len(set(sentence.split())) / len(sentence.split()) if len(sentence.split()) > 0 else 0,
    }

def remove_personal_names(sentences):
    SKIP_PHRASES = {
        "Research Division", "Equity Research", "Capital Markets", "Investment Officer",
        "Vice Chairman", "Managing Director", "Portfolio Manager", "Chief Investment Officer",
        "Chief Financial Officer", "Chief Operating Officer", "President", "CEO", "Chairman",
        "Senior Analyst", "Financial Analyst", "BMO Capital", "RBC Capital", "Morgan Stanley", "Goldman Sachs"
    }

    batch_size = 50  # ✅ Process in chunks of 50 sentences
    filtered_sentences = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        docs = list(nlp.pipe(batch))  
        for sent, doc in zip(batch, docs):
            if not any(ent.label_ == "PERSON" for ent in doc.ents) and not any(phrase in sent for phrase in SKIP_PHRASES):
                filtered_sentences.append(sent)

    return filtered_sentences

def analyze_sentiment_vader(sentence):
    scores = vader.polarity_scores(sentence)
    return scores["pos"], scores["neg"], scores["neu"]

# def analyze_sentiment_finbert(sentence):
#     try:
#         finbert_result = finbert(sentence)
#         if isinstance(finbert_result, list) and len(finbert_result) > 0 and isinstance(finbert_result[0], list):
#             finbert_result = finbert_result[0]

#         sentiment_dict = {score["label"].lower(): score["score"] for score in finbert_result}
#         return sentiment_dict.get("positive", 0), sentiment_dict.get("negative", 0), sentiment_dict.get("neutral", 0)
#     except Exception as e:
#         return 0, 0, 0
    
def analyze_sentiment_finbert(sentence):
    try:
        tokens = tokenizer.tokenize(sentence)

        if len(tokens) > 512:
            logging.warning(f"⚠️ Sentence too long ({len(tokens)} tokens). Truncating...")
            tokens = tokens[:512]  # ✅ Keep only first 512 tokens
            sentence = tokenizer.convert_tokens_to_string(tokens)  # ✅ Convert back to text
        
        finbert_result = finbert(sentence)
        sentiment_dict = {score["label"].lower(): score["score"] for score in finbert_result}
        
        return sentiment_dict.get("positive", 0), sentiment_dict.get("negative", 0), sentiment_dict.get("neutral", 0)

    except Exception as e:
        logging.error(f"❌ FinBERT Error: {e}")
        return 0, 0, 0

def analyze_file(file_path, file_index, total_files):
    """ Extracts text, ticker, year, and quarter, and runs sentiment analysis. """
    logging.info(f"🟢 START Processing {file_index}/{total_files}: {file_path}")

    file_name = os.path.basename(file_path)
    ticker, year, quarter = extract_ticker_year_quarter(file_name)

    if ticker is None or year is None or quarter is None:
        logging.warning(f"⚠️ Skipping {file_name} (Invalid filename format)")
        return file_name, []

    # Use the correct extraction method based on `sentiment_flag`
    if sentiment_flag == "sell_side":
        text = extract_text_from_txt(file_path)  # Sell-side reports use .txt
    else:
        text = extract_text_from_pdf(file_path)  # Transcripts use .pdf

    if not text.strip():
        logging.warning(f"⚠️ Empty text extracted from {file_name}")
        return file_name, []

    logging.info(f"🏢 Extracted ticker: {ticker}, Year: {year}, Quarter: {quarter}")

    keyword_category_map = {kw: "Financial" for kw in keywords_financial}
    keyword_category_map.update({kw: "Climate" for kw in keywords_climate})
    keyword_category_map.update({kw: "Risk" for kw in keywords_risk})

    sentences = [sent.text for sent in nlp(text).sents]
    logging.info(f"✂️ Extracted {len(sentences)} sentences from {file_name}")

    filtered_sentences_info = [
        (sent, kw, keyword_category_map[kw])
        for sent in sentences
        for kw in keyword_category_map.keys()
        if kw in sent.lower()
    ]

    if not filtered_sentences_info:
        logging.warning(f"⚠️ No relevant sentences found in {file_name}")
        return file_name, []

    results = []
    for sentence, keyword, category in filtered_sentences_info:
        finbert_pos, finbert_neg, finbert_neu = analyze_sentiment_finbert(sentence)
        vader_pos, vader_neg, vader_neu = analyze_sentiment_vader(sentence)
        readability = calculate_readability(sentence)

        results.append({
            "file_name": file_name,
            "Ticker": ticker,
            "Year": year,
            "Quarter": quarter,
            "sentence": sentence,
            "keyword": keyword,
            "category": category,
            "finbert_positive": finbert_pos,
            "finbert_negative": finbert_neg,
            "finbert_neutral": finbert_neu,
            "vader_positive": vader_pos,
            "vader_negative": vader_neg,
            "vader_neutral": vader_neu,
            "flesch_reading_ease": readability["flesch_reading_ease"],
            "gunning_fog_index": readability["gunning_fog_index"],
            "smog_index": readability["smog_index"],
            "lexical_density": readability["lexical_density"],
            "word_count": len(sentence.split()),
        })

    logging.info(f"✅ Finished processing {file_name} with {len(results)} results")
    return file_name, results

def process_batch(batch_number):
    logging.info(f"🚀 Starting batch {batch_number}")

    # Select the correct folder based on `sentiment_flag`
    if sentiment_flag == "sell_side":
        data_folder = SELL_SIDE_DATA
        file_extension = ".txt"
    else:
        data_folder = TRANSCRIPT_DATA
        file_extension = ".pdf"

    all_files = sorted([
        os.path.join(data_folder, f) for f in os.listdir(data_folder)
        if f.endswith(file_extension)
    ])

    start_idx = (batch_number - 1) * NUM_PDFS
    end_idx = start_idx + NUM_PDFS
    batch_files = all_files[start_idx:end_idx]

    if not batch_files:
        logging.warning(f"⚠️ No files found for batch {batch_number}")
        return

    all_results = []
    for file_index, file_path in enumerate(batch_files, start=1):
        logging.debug(f"🟢 Processing {file_index}/{len(batch_files)}: {file_path}")
        try:
            file_name, results = analyze_file(file_path, file_index, len(batch_files))
            all_results.extend(results)
            logging.info(f"✅ Finished {file_name} ({len(results)} results)")
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"❌ Error processing {file_path}: {e}\n{error_details}")

    batch_output_file = os.path.join(OUTPUT_FOLDER, f"sentiment_results_batch_{batch_number}.csv")
    pd.DataFrame(all_results).to_csv(batch_output_file, index=False)
    logging.info(f"✅ Batch {batch_number} saved successfully")

def merge_batches():
    """Merges all sentiment batch files (including PGR data) into one final CSV."""
    batch_files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("sentiment_results_batch_")])

    if sentiment_flag == "sell_side":
        pgr_file = os.path.join(OUTPUT_FOLDER, "pgr_sentiment_results.csv")
        if os.path.exists(pgr_file):
            batch_files.append("pgr_sentiment_results.csv")  # ✅ Include PGR data in merge

    if not batch_files:
        logging.warning("⚠️ No batch files found for merging.")
        return

    df_list = []
    for f in batch_files:
        file_path = os.path.join(OUTPUT_FOLDER, f)
        
        # ✅ Skip empty files
        if os.path.getsize(file_path) == 0:
            logging.warning(f"⚠️ Skipping empty CSV file: {f}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logging.warning(f"⚠️ Skipping empty dataframe: {f}")
                continue
            df_list.append(df)
        except Exception as e:
            logging.error(f"❌ Error reading {f}: {e}")
            continue

    if not df_list:
        logging.error("⚠️ No valid batch files found for merging.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)

    # ✅ Save the merged sentiment results
    final_output_file = os.path.join(OUTPUT_FOLDER, f"{sentiment_flag}_sentiment_results_merged.csv")
    merged_df.to_csv(final_output_file, index=False)
    logging.info(f"✅ Merged sentiment results saved to {final_output_file}")

def analyze_sentiment_trends(df):
    sentiment_trends = df.groupby(["Year", "Quarter"]).agg(
        avg_finbert=("finbert_positive", "mean"),
        avg_vader=("vader_positive", "mean")
    ).reset_index()
    sentiment_trends.to_csv(os.path.join(ANALYSIS_FOLDER, f"{sentiment_flag}_sentiment_trends.csv"), index=False)

def analyze_keyword_sentiment(df):
    keyword_analysis = df.groupby("keyword").agg(
        count=("keyword", "count"),
        avg_finbert=("finbert_positive", "mean"),
        avg_vader=("vader_positive", "mean")
    ).reset_index()
    keyword_analysis.to_csv(os.path.join(KEYWORD_FOLDER, f"{sentiment_flag}_keyword_sentiment.csv"), index=False)

def extract_month_year_from_filename(filename):
    """ Extracts month and year from a filename like 'PGR_JAN25.pdf' """
    try:
        parts = filename.split("_")
        if len(parts) < 2:
            return None, None  # Invalid format
        
        month_str = parts[1][:3].upper()  # Extract first 3 letters (e.g., 'JAN')
        year_str = "20" + parts[1][3:5]  # Extract last 2 digits (e.g., '25' → '2025')

        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        
        month = month_map.get(month_str, None)
        year = int(year_str)

        return month, year
    except Exception as e:
        logging.error(f"Error extracting month/year from filename {filename}: {e}")
        return None, None

def process_pgr_files():
    """Processes all PGR files in the sell-side folder, extracts data, and converts to quarterly averages."""
    logging.info("🚀 Processing PGR files in sell-side folder")

    pgr_files = [f for f in os.listdir(SELL_SIDE_DATA) if f.startswith("PGR_") and f.endswith(".txt")]

    if not pgr_files:
        logging.warning("⚠️ No PGR files found in sell-side folder.")
        return

    # ✅ Define keyword-category mappings
    keyword_category_map = {kw: "Financial" for kw in keywords_financial}
    keyword_category_map.update({kw: "Climate" for kw in keywords_climate})
    keyword_category_map.update({kw: "Risk" for kw in keywords_risk})

    all_results = []

    for file in pgr_files:
        file_path = os.path.join(SELL_SIDE_DATA, file)
        text = extract_text_from_txt(file_path)

        ticker, year, quarter = extract_ticker_year_quarter(file)
        
        if ticker is None or year is None or quarter is None:
            logging.warning(f"⚠️ Skipping {file} (Invalid filename format)")
            continue

        try:
            # ✅ Extract sentences
            sentences = [sent.text for sent in nlp(text).sents]
            
            # ✅ Filter sentences containing relevant keywords
            filtered_sentences_info = [
                (sent, kw, keyword_category_map[kw])
                for sent in sentences
                for kw in keyword_category_map.keys()
                if kw in sent.lower()
            ]

            if not filtered_sentences_info:
                logging.warning(f"⚠️ No relevant sentences found in {file}")
                continue

            # ✅ Run analysis on filtered sentences
            for sentence, keyword, category in filtered_sentences_info:
                finbert_pos, finbert_neg, finbert_neu = analyze_sentiment_finbert(sentence)
                vader_pos, vader_neg, vader_neu = analyze_sentiment_vader(sentence)
                readability = calculate_readability(sentence)

                all_results.append({
                    "file_name": file,
                    "Ticker": ticker,
                    "Year": year,
                    "Quarter": quarter,
                    "sentence": sentence,
                    "keyword": keyword,
                    "category": category,
                    "finbert_positive": finbert_pos,
                    "finbert_negative": finbert_neg,
                    "finbert_neutral": finbert_neu,
                    "vader_positive": vader_pos,
                    "vader_negative": vader_neg,
                    "vader_neutral": vader_neu,
                    "flesch_reading_ease": readability["flesch_reading_ease"],
                    "gunning_fog_index": readability["gunning_fog_index"],
                    "smog_index": readability["smog_index"],
                    "lexical_density": readability["lexical_density"],
                    "word_count": len(sentence.split()),
                })

        except Exception as e:
            logging.error(f"❌ Error processing PGR file {file}: {e}\n{traceback.format_exc()}")

    # ✅ Save PGR results in the same format as `process_batch()`
    pgr_output_file = os.path.join(OUTPUT_FOLDER, "pgr_sentiment_results.csv")
    pd.DataFrame(all_results).to_csv(pgr_output_file, index=False)
    logging.info(f"✅ PGR Sentiment results saved to {pgr_output_file}")

def perform_analysis():
    merged_csv_path = os.path.join(OUTPUT_FOLDER, f"{sentiment_flag}_sentiment_results_merged.csv")  

    if not os.path.exists(merged_csv_path):
        logging.error(f"File not found: {merged_csv_path}")
        return  

    merged_df = pd.read_csv(merged_csv_path)

    analyze_sentiment_trends(merged_df)
    analyze_keyword_sentiment(merged_df)

def generate_sentiment_excel(data_folder=f"data/{sentiment_flag}_output_gpt", output_path=f"output/{sentiment_flag}_sentiment_analysis_gpt.xlsx"):
    
    file_path = os.path.join(data_folder, f"{sentiment_flag}_sentiment_results_merged_gpt.csv")

    sentiment_df = pd.read_csv(file_path)
    
    sentiment_df.columns = sentiment_df.columns.str.upper()
    sentiment_df["YEAR"] = sentiment_df["YEAR"].astype(int)
    
    sentiment_agg = sentiment_df.groupby(["TICKER", "YEAR", "CATEGORY"], as_index=False).agg(
        AVG_FINBERT=("FINBERT_NEGATIVE", "mean"),
        # AVG_VADER=("VADER_NEGATIVE", "mean")
    )
    
    overall_sentiment = sentiment_df.groupby(["TICKER", "YEAR"], as_index=False).agg(
        AVG_FINBERT=("FINBERT_NEGATIVE", "mean"),
        # AVG_VADER=("VADER_NEGATIVE", "mean")
    )
    
    grouped_data = {}
    keyword_types = sentiment_df["CATEGORY"].unique()
    # sentiment_models = ["FINBERT", "VADER"]
    sentiment_models = ['FINBERT']
    
    for keyword in keyword_types:
        for model in sentiment_models:
            sentiment_column = f"AVG_{model}"
            df_pivot = sentiment_agg[sentiment_agg["CATEGORY"] == keyword].pivot(
                index="TICKER", columns="YEAR", values=sentiment_column
            ).fillna(0)
            grouped_data[f"{keyword}_{model}"] = df_pivot
    
    grouped_data["Overall_FINBERT"] = overall_sentiment.pivot(index="TICKER", columns="YEAR", values="AVG_FINBERT").fillna(0)
    # grouped_data["Overall_VADER"] = overall_sentiment.pivot(index="TICKER", columns="YEAR", values="AVG_VADER").fillna(0)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, df in grouped_data.items():
            df.to_excel(writer, sheet_name=sheet_name)

def compute_quarterly_negative_sentiment(input_folder=f"data/{sentiment_flag}_output", output_folder="analysis/sentiment"):
    """
    Computes the average quarterly negative sentiment for both 'climate' and 'all' sentiment categories.
    Works for all sentiment flags and accounts for PGR's different formatting.
    """
    input_file = os.path.join(input_folder, f"{sentiment_flag}_sentiment_results_merged.csv")
    climate_output_file = os.path.join(output_folder, f"{sentiment_flag}_quarterly_climate_sentiment.csv")
    all_output_file = os.path.join(output_folder, f"{sentiment_flag}_quarterly_all_sentiment.csv")
    
    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return  
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.upper()
    
    # Ensure correct data types
    df["YEAR"] = df["YEAR"].astype(int)
    df["QUARTER"] = df["QUARTER"].astype(str)
    
    # Compute sentiment for climate category
    df_climate = df[df["CATEGORY"] == "Climate"]
    sentiment_quarterly_climate = df_climate.groupby(["TICKER", "YEAR", "QUARTER"], as_index=False).agg(
        AVG_FINBERT_NEGATIVE=("FINBERT_NEGATIVE", "mean"),
        AVG_VADER_NEGATIVE=("VADER_NEGATIVE", "mean")
    )
    
    # Compute sentiment for all categories
    sentiment_quarterly_all = df.groupby(["TICKER", "YEAR", "QUARTER"], as_index=False).agg(
        AVG_FINBERT_NEGATIVE=("FINBERT_NEGATIVE", "mean"),
        AVG_VADER_NEGATIVE=("VADER_NEGATIVE", "mean")
    )
    
    # Save results
    os.makedirs(output_folder, exist_ok=True)
    sentiment_quarterly_climate.to_csv(climate_output_file, index=False)
    sentiment_quarterly_all.to_csv(all_output_file, index=False)
    logging.info(f"✅ Quarterly climate sentiment saved to {climate_output_file}")
    logging.info(f"✅ Quarterly all sentiment saved to {all_output_file}")
    
    return sentiment_quarterly_climate, sentiment_quarterly_all

if __name__ == "__main__":
    load_finbert()
    process_pgr_files()
    process_batch(batch_number)
    merge_batches()
    perform_analysis()
    generate_sentiment_excel()
    compute_quarterly_negative_sentiment()