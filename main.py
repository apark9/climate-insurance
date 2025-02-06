import logging
import time
import os
import code.sentiment as sentiment
import code.graphing as graphing
import code.climate as climate
import code.financials as financials
import code.keywords as keywords
import code.naic as naic
import sys

def setup_logging(job_name):
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{job_name}_{timestamp}.log"

    # Remove any existing handlers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)  # ✅ Ensures logs appear in the terminal
        ]
    )

    # Force logs to flush immediately
    logging.info(f"📂 Logging started: {log_file}")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    # try:
    #     setup_logging("keywords")
    #     keywords.analyze_keywords()
    # except Exception as e:
    #     logging.error(f"Error in keywords analysis: {e}")

    try:
        setup_logging("NAIC disclosures")
        naic.analyze_disclosures()
    except Exception as e:
        logging.error(f"Error in keywords analysis: {e}")

    # try:
    #     setup_logging("sentiment")
    #     sentiment.analyze_transcripts()
    # except Exception as e:
    #     logging.error(f"Error in sentiment analysis: {e}")

    # try:
    #     setup_logging("climate")
    #     climate.run_climate_analysis()
    # except Exception as e:
    #     logging.error(f"Error in graphing: {e}")

    # try:
    #     setup_logging("financials")
    #     financials.run_financial_analysis()
    #     financials.run_models()
    # except Exception as e:
    #     logging.error(f"Error in financial analysis: {e}")

    # try:
    #     setup_logging("graphing")
    #     graphing.perform_graphing()
    # except Exception as e:
    #     logging.error(f"Error in graphing: {e}")