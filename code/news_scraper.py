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

def is_relevant_article(title, snippet, company_name):
    """Check if the article title or snippet contains the company name."""
    company_name_lower = company_name.lower()
    return company_name_lower in title.lower() or company_name_lower in snippet.lower()

def increment_request_count():
    global api_request_count
    api_request_count += 1
    if api_request_count >= DAILY_LIMIT:
        logging.info(f"Daily API limit reached ({DAILY_LIMIT} requests). Sleeping until the next day.")
        # Sleep until midnight
        now = datetime.now()
        midnight = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        sleep_time = (midnight - now).seconds
        time.sleep(sleep_time)
        api_request_count = 0  # Reset request count for the new day
        return False
    return True

def search_google(query, company_name, num_results=10, retries=3):
    global api_request_count
    results = []
    if api_request_count >= DAILY_LIMIT:
        logging.warning("Daily API limit reached. Skipping further requests.")
        return results
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&num={num_results}&lr=lang_en"
        logging.info(f"Fetching URL: {url}")
        response = requests.get(url)
        increment_request_count()
        if response.status_code == 429:
            logging.error(f"Rate limit hit: {url}. Retrying...")
            for i in range(retries):
                time.sleep((2 ** i) * REQUEST_DELAY)
                response = requests.get(url)
                increment_request_count()
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
            if is_relevant_article(title, snippet, company_name):
                logging.info(f"Relevant Article: {title} - {link}")
                results.append({"title": title, "link": link, "snippet": snippet, "date": date})
            else:
                logging.info(f"Irrelevant Article Skipped: {title} - {link}")
    except Exception as e:
        logging.error(f"Error during API call: {e}")
    return results

def collect_news(for_month=None, for_year=None, log_file="news_scraper.log"):
    global api_request_count
    setup_logging(log_file)
    all_results = {}
    timestamp = datetime.now().strftime("%Y_%B")
    if for_month and for_year:
        timestamp = f"{for_year}_{for_month}"

    output_folder = os.path.abspath("output/news")
    os.makedirs(output_folder, exist_ok=True)

    for company, keywords in company_keywords.items():
        logging.info(f"Scraping articles for company: {company}")
        company_results = []
        for keyword in keywords:
            if not increment_request_count():
                return
            company_results.extend(search_google(keyword, company))
            time.sleep(REQUEST_DELAY)
        all_results[company] = company_results

    logging.info("Scraping articles for general keywords")
    general_results = []
    for keyword in general_keywords:
        if not increment_request_count():
            return  # Stop scraping for the day if limit is reached
        general_results.extend(search_google(keyword, ""))
        time.sleep(REQUEST_DELAY)
    all_results["General"] = general_results

    for category, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            output_file = f"{output_folder}/news_results_{category}_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logging.info(f"Saved results for {category} to {output_file}")
        else:
            logging.warning(f"No results found for {category}")

def save_progress(year, month):
    with open("progress.txt", "w") as f:
        f.write(f"{year},{month}")

def load_progress():
    if os.path.exists("progress.txt"):
        with open("progress.txt", "r") as f:
            year, month = map(int, f.read().strip().split(","))
            return year, month
    return 2007, 1  # Default start year and month


def run_past_months():
    global api_request_count
    setup_logging("past_months.log")
    start_year, start_month = load_progress()
    end_year, end_month = 2024, 1

    current_year, current_month = start_year, start_month
    while (current_year, current_month) <= (end_year, end_month):
        if api_request_count >= DAILY_LIMIT:
            logging.info("Stopping past months scrape for today due to API limit.")
            save_progress(current_year, current_month)
            break
        month_name_str = month_name[current_month]
        logging.info(f"Collecting news for {month_name_str} {current_year}")
        collect_news(for_month=month_name_str, for_year=current_year, log_file="past_months.log")
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
    save_progress(current_year, current_month)

def run_current_news():
    setup_logging("current_news.log")
    logging.info("Starting current news scrape.")
    collect_news()

def run_current_news_monthly():
    setup_logging("current_news_monthly.log")
    if datetime.now().day == 1:
        logging.info("Running current news scrape for the first of the month.")
        run_current_news()
    else:
        logging.info("Not the first of the month. Skipping current news scrape.")

schedule.every().day.at("02:00").do(run_past_months)
schedule.every().day.at("02:00").do(run_current_news_monthly)

if __name__ == "__main__":
    setup_logging("main")
    logging.info("Starting continuous scraping process.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)  # Retry after 1 minute
