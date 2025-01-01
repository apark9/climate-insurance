import requests
import pandas as pd
import time
import logging
from datetime import datetime
from calendar import month_name
import os

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )

company_keywords = {
    "Progressive": ['"Progressive insurance"', '"Progressive climate risk"'],
    "AllState": ['"AllState insurance"', '"AllState reinsurance"'],
    "Aflac": ['"Aflac insurance"', '"Aflac reinsurance challenges"'],
    "American Financial Group": ['"American Financial Group insurance"', '"AFG climate risk"'],
    "Travelers": ['"Travelers insurance"', '"Travelers exiting states"']
}

general_keywords = [
    '"insurers exiting states"',
    '"climate risk insurance adaptation"',
    '"reinsurance challenges"',
    '"climate risk affecting insurers"',
    '"insurance climate exit"'
]

API_KEY = "AIzaSyAT9oXOGzyP1B0Gec_OhbwwtO2AUY6p1E8"
CX = "c70ecdcc4723d40f0"
REQUEST_DELAY = 2

def search_google(query, num_results=10):
    results = []
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&num={num_results}"
        logging.info(f"Fetching URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch {url}: {response.status_code}")
            return results
        data = response.json()
        for item in data.get("items", []):
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")
            logging.info(f"Scraped Article: {title} - {link}")
            results.append({"title": title, "link": link, "snippet": snippet})
    except Exception as e:
        logging.error(f"Error during API call: {e}")
    return results

def collect_news(for_month=None, for_year=None, log_file="news_scraper.log"):
    setup_logging(log_file)
    all_results = {}
    timestamp = datetime.now().strftime("%Y_%B")
    if for_month and for_year:
        timestamp = f"{for_year}_{for_month}"

    for company, keywords in company_keywords.items():
        logging.info(f"Scraping articles for company: {company}")
        company_results = []
        for keyword in keywords:
            company_results.extend(search_google(keyword))
            time.sleep(REQUEST_DELAY)
        all_results[company] = company_results
    
    logging.info("Scraping articles for general keywords")
    general_results = []
    for keyword in general_keywords:
        general_results.extend(search_google(keyword))
        time.sleep(REQUEST_DELAY)
    all_results["General"] = general_results
    
    output_folder = "output/news"
    os.makedirs(output_folder, exist_ok=True)
    for category, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            output_file = f"{output_folder}/news_results_{category}_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logging.info(f"Saved results for {category} to {output_file}")
        else:
            logging.warning(f"No results found for {category}")

def run_past_months():
    setup_logging("past_months.log")
    start_year, start_month = 2007, 1
    end_year, end_month = 2024, 1

    current_year, current_month = start_year, start_month
    while (current_year, current_month) <= (end_year, end_month):
        month_name_str = month_name[current_month]
        logging.info(f"Collecting news for {month_name_str} {current_year}")
        collect_news(for_month=month_name_str, for_year=current_year, log_file="past_months.log")
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1

if __name__ == "__main__":
    # collect_news()
    run_past_months()
