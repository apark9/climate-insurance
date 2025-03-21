import os
import json
import pandas as pd
import spacy
from collections import Counter, defaultdict
from PyPDF2 import PdfReader
from multiprocessing import Pool

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 4_000_000

TRANSCRIPT_FOLDER = "data/transcripts"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

COMPANY_KEYWORDS_FILE = os.path.join(OUTPUT_FOLDER, "company_keywords.json")
GLOBAL_KEYWORDS_FILE = os.path.join(OUTPUT_FOLDER, "global_keywords.json")
KEYWORD_CSV_FILE = os.path.join(OUTPUT_FOLDER, "keyword_frequencies.csv")

CHUNK_SIZE = 500_000


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        return ""


def parse_filename(file_name):
    try:
        parts = file_name.split("_")
        return parts[0]
    except Exception as e:
        return "UNKNOWN"


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_most_common_words(text, top_n=50):
    if not text:
        return []

    try:
        docs = list(nlp.pipe([text], batch_size=5, disable=["parser", "ner"]))
        words = [token.text.lower() for token in docs[0] if token.is_alpha and not token.is_stop]
        return Counter(words).most_common(top_n)
    except Exception as e:
        return []


def process_file(file_path):
    file_name = os.path.basename(file_path)
    company = parse_filename(file_name)

    text = extract_text_from_pdf(file_path)
    if not text:
        return company, file_name, []

    if len(text) > nlp.max_length:
        text_chunks = split_text_into_chunks(text)
    else:
        text_chunks = [text]

    keywords = []
    for chunk in text_chunks:
        keywords.extend(get_most_common_words(chunk, top_n=50))

    aggregated_keywords = Counter(dict(keywords)).most_common(50)
    return company, file_name, aggregated_keywords


def analyze_keywords():
    input_files = [os.path.join(TRANSCRIPT_FOLDER, f) for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".pdf")]

    if not input_files:
        return

    per_file_keywords = {}
    per_company_keywords = defaultdict(Counter)
    global_keywords_counter = Counter()

    with Pool(processes=8) as pool:
        results = pool.map(process_file, input_files)

    for company, file_name, keywords in results:
        if keywords:
            per_file_keywords[file_name] = keywords
            per_company_keywords[company].update(dict(keywords))
            global_keywords_counter.update(dict(keywords))

    per_company_keywords = {
        company: counter.most_common(50) for company, counter in per_company_keywords.items()
    }
    global_keywords = global_keywords_counter.most_common(50)

    with open(COMPANY_KEYWORDS_FILE, "w") as f:
        json.dump(per_company_keywords, f, indent=4)

    with open(GLOBAL_KEYWORDS_FILE, "w") as f:
        json.dump(global_keywords, f, indent=4)

    keyword_list = []
    for company, words in per_company_keywords.items():
        for word, count in words:
            keyword_list.append({"company": company, "word": word, "count": count})

    global_keyword_list = [{"company": "ALL", "word": word, "count": count} for word, count in global_keywords]
    keyword_list.extend(global_keyword_list)

    keyword_df = pd.DataFrame(keyword_list)
    keyword_df.to_csv(KEYWORD_CSV_FILE, index=False)

if __name__ == "__main__":
    analyze_keywords()