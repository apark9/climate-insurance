import os
import logging
import pandas as pd
import spacy
from PyPDF2 import PdfReader
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from textstat import textstat
from config import keywords_financial, keywords_climate, keywords_risk
from nltk.sentiment import SentimentIntensityAnalyzer

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/sentiment_analysis.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

nlp = spacy.load("en_core_web_sm")
finbert = None
vader = SentimentIntensityAnalyzer()

TRANSCRIPT_FOLDER = "data/transcripts"
OUTPUT_FOLDER = "data/sentiment_output"
ANALYSIS_FOLDER = "analysis/sentiment"

NUM_WORKERS = 4
NUM_PDFS = 100
MAX_TOKENS = 512
TOTAL_BATCHES = 21

def load_finbert():
    global finbert
    if finbert is None:
        finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", return_all_scores=True)
        logging.info("FinBERT loaded.")

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

def extract_ticker(file_name, text):
    try:
        ticker = file_name.split("_")[0]  

        first_lines = text.split("\n")[:10]
        for line in first_lines:
            cleaned_line = line.strip()

            if any(word in cleaned_line.lower() for word in ["copyright", "all rights reserved", "sp global", "intelligence"]):
                continue

            if any(exchange in cleaned_line for exchange in ["NYSE:", "NASDAQ:", "TSX:", "LSE:", "ASX:"]):
                continue

            return ticker

        return ticker
    except Exception as e:
        logging.error(f"Error extracting ticker from {file_name}: {e}")
        return file_name.split("_")[0]

def calculate_readability(sentence):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(sentence),
        "gunning_fog_index": textstat.gunning_fog(sentence),
        "smog_index": textstat.smog_index(sentence),
        "lexical_density": len(set(sentence.split())) / len(sentence.split()) if len(sentence.split()) > 0 else 0,
    }

def remove_personal_names(sentences):
    SKIP_PHRASES = [
        "Research Division", "Equity Research", "Capital Markets", "Investment Officer",
        "Vice Chairman", "Managing Director", "Portfolio Manager", "Chief Investment Officer",
        "Chief Financial Officer", "Chief Operating Officer", "President", "CEO", "Chairman",
        "Senior Analyst", "Financial Analyst", "BMO Capital", "RBC Capital", "Morgan Stanley", "Goldman Sachs"
    ]
    
    filtered_sentences = []
    for sent in sentences:
        doc = nlp(sent)
        if any(ent.label_ == "PERSON" for ent in doc.ents):
            continue  
        if any(phrase in sent for phrase in SKIP_PHRASES):
            continue  
        filtered_sentences.append(sent)

    return filtered_sentences

def analyze_sentiment_vader(sentence):
    scores = vader.polarity_scores(sentence)
    return scores["pos"], scores["neg"], scores["neu"]

def analyze_sentiment_finbert(sentence):
    try:
        finbert_result = finbert(sentence)
        if isinstance(finbert_result, list) and len(finbert_result) > 0 and isinstance(finbert_result[0], list):
            finbert_result = finbert_result[0]

        sentiment_dict = {score["label"].lower(): score["score"] for score in finbert_result}
        return sentiment_dict.get("positive", 0), sentiment_dict.get("negative", 0), sentiment_dict.get("neutral", 0)
    except Exception as e:
        return 0, 0, 0

def analyze_pdf(file_path, file_index, total_files):
    file_name = os.path.basename(file_path)
    ticker, year, quarter = extract_date_from_filename(file_name)

    if ticker is None or year is None or quarter is None:
        return file_name, []

    logging.info(f"Processing {file_index}/{total_files}: {file_name}")

    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return file_name, []

    company_name = extract_ticker(file_name, text)

    keyword_category_map = {kw: "Financial" for kw in keywords_financial}
    keyword_category_map.update({kw: "Climate" for kw in keywords_climate})
    keyword_category_map.update({kw: "Risk" for kw in keywords_risk})

    filtered_sentences_info = []
    for sent in nlp(text).sents:
        sent_text = sent.text
        for kw in keyword_category_map.keys():
            if kw in sent_text.lower():
                filtered_sentences_info.append((sent_text, kw, keyword_category_map[kw]))
                break 
            
    filtered_sentences_info = [(sent, kw, cat) for sent, kw, cat in filtered_sentences_info if sent in remove_personal_names([sent])]

    if not filtered_sentences_info:
        return file_name, []

    load_finbert()

    results = []
    for sentence, keyword, category in filtered_sentences_info:
        finbert_pos, finbert_neg, finbert_neu = analyze_sentiment_finbert(sentence)
        vader_pos, vader_neg, vader_neu = analyze_sentiment_vader(sentence)
        readability = calculate_readability(sentence)
        doc = nlp(sentence)
        
        results.append({
            "file_name": file_name,
            "Ticker": company_name,
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
            "named_entity_count": len(doc.ents),
        })

    return file_name, results

def process_batch(batch_number):
    all_files = sorted([os.path.join(TRANSCRIPT_FOLDER, f) for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".pdf")])
    start_idx = (batch_number - 1) * NUM_PDFS
    end_idx = start_idx + NUM_PDFS
    batch_files = all_files[start_idx:end_idx]

    if not batch_files:
        return

    all_results = []
    for file_index, file_path in enumerate(batch_files, start=1):
        file_name, results = analyze_pdf(file_path, file_index, len(batch_files))
        all_results.extend(results)

    batch_output_file = os.path.join(OUTPUT_FOLDER, f"sentiment_results_batch_{batch_number}.csv")
    pd.DataFrame(all_results).to_csv(batch_output_file, index=False)

def merge_batches():
    batch_files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("sentiment_results_batch_")])
    if not batch_files:
        return

    df_list = [pd.read_csv(os.path.join(OUTPUT_FOLDER, f)) for f in batch_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(os.path.join(OUTPUT_FOLDER, "sentiment_results_merged.csv"), index=False)

def analyze_sentiment_trends(df):
    sentiment_trends = df.groupby(["Year", "Quarter"]).agg(
        avg_finbert=("finbert_positive", "mean"),
        avg_vader=("vader_positive", "mean")
    ).reset_index()
    sentiment_trends.to_csv(os.path.join(ANALYSIS_FOLDER, "sentiment_trends.csv"), index=False)

def analyze_keyword_sentiment(df):
    keyword_analysis = df.groupby("keyword").agg(
        count=("keyword", "count"),
        avg_finbert=("finbert_positive", "mean"),
        avg_vader=("vader_positive", "mean")
    ).reset_index()
    keyword_analysis.to_csv(os.path.join(ANALYSIS_FOLDER, "keyword_sentiment.csv"), index=False)

def perform_analysis():
    merged_csv_path = os.path.join(OUTPUT_FOLDER, "sentiment_results_merged.csv")  

    if not os.path.exists(merged_csv_path):
        logging.error(f"File not found: {merged_csv_path}")
        return  

    merged_df = pd.read_csv(merged_csv_path)

    analyze_sentiment_trends(merged_df)
    analyze_keyword_sentiment(merged_df)

if __name__ == "__main__":
    process_batch(batch_number)
    merge_batches()
    perform_analysis()
