import logging
import time
import os
import code.sentiment as sentiment
import code.graphing as graphing
import code.climate as climate
import code.financials as financials
import code.keywords as keywords
import os

def setup_logging(job_name="main"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{job_name}_{timestamp}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    os.makedirs("logs", exist_ok=True)  # Ensure the logs folder exists
    logging.info(f"Log setup complete for {job_name}")

if __name__ == "__main__":
    try:
        setup_logging("keywords")
        keywords.analyze_keywords()
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