import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from calendar import month_name
from config import API_KEY, CX, company_keywords, general_keywords
import os
import schedule

REQUEST_DELAY = 30
DAILY_LIMIT = 100
api_request_count = 0
progress_file = "news_progress.json"
output_folder = "output/news"

def setup_logging(job_name="main"):
    log_dir = os.path.abspath("logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{job_name}_{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logging.info(f"Log setup complete for {job_name}")

def load_progress():
    if os.path.exists(progress_file):
        return pd.read_json(progress_file)
    return pd.DataFrame(columns=["category", "keyword", "status"])

def save_progress(progress_df):
    progress_df.to_json(progress_file, orient="records")

def increment_request_count():
    global api_request_count
    api_request_count += 1
    if api_request_count >= DAILY_LIMIT:
        logging.info(f"Daily API limit reached ({DAILY_LIMIT} requests). Stopping for today.")
        return False
    return True

def search_google(query, category, num_results=10, retries=3):
    global api_request_count
    results = []
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&num={num_results}&lr=lang_en"
        logging.info(f"Fetching URL: {url}")
        response = requests.get(url)
        if not increment_request_count():
            return results
        if response.status_code == 429:
            logging.error(f"Rate limit hit: {url}. Retrying...")
            for i in range(retries):
                time.sleep((2 ** i) * REQUEST_DELAY)
                response = requests.get(url)
                if response.status_code == 200:
                    break
            else:
                logging.error(f"Failed after retries: {url}")
                return results
        if response.status_code != 200:
            logging.error(f"Failed to fetch {url}: {response.status_code}")
            return results
        data = response.json()
        for item in data.get("items", []):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            date = item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", datetime.now().strftime("%Y-%m-%d"))
            results.append({"title": title, "link": link, "snippet": snippet, "date": date, "category": category})
    except Exception as e:
        logging.error(f"Error during API call: {e}")
    return results

def collect_news():
    global api_request_count
    progress_df = load_progress()
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%B")

    for category, keywords in {**company_keywords, "General": general_keywords}.items():
        for keyword in keywords:
            if not increment_request_count():
                save_progress(progress_df)
                return
            if not progress_df[(progress_df["category"] == category) & (progress_df["keyword"] == keyword)].empty:
                logging.info(f"Skipping completed keyword: {keyword} in category: {category}")
                continue
            results = search_google(keyword, category)
            time.sleep(REQUEST_DELAY)
            if results:
                df = pd.DataFrame(results)
                output_file = f"{output_folder}/news_results_{category}_{timestamp}.csv"
                df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
                logging.info(f"Saved results for keyword: {keyword} in category: {category} to {output_file}")
            progress_df = progress_df.append({"category": category, "keyword": keyword, "status": "completed"}, ignore_index=True)
    save_progress(progress_df)

if __name__ == "__main__":
    setup_logging("main")
    logging.info("Starting news scraping with progress tracking.")
    collect_news()
