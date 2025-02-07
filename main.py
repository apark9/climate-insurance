import logging
import time
import os
import sys
import code.sentiment as sentiment
import code.graphing as graphing
import code.climate as climate
import code.financials as financials
import code.keywords as keywords
import code.naic as naic

def setup_logging():
    """Sets up logging with real-time updates."""
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # ✅ Only timestamp in log file name
    log_file = f"logs/{timestamp}.log"

    # ✅ Remove existing handlers to prevent duplicate logs
    if logging.root.handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)  # ✅ Logs show in both terminal & file
        ]
    )

    logging.info(f"📂 Logging started: {log_file}")
    sys.stdout.flush()
    sys.stderr.flush()

    return log_file

if __name__ == "__main__":
    log_file = setup_logging()  # ✅ Set up logging once with a timestamped file

    try:
        logging.info("📝 Running NAIC disclosures analysis...")
        naic.analyze_disclosures()
    except Exception as e:
        logging.error(f"❌ Error in NAIC disclosures: {e}")

    # Uncomment these to include additional processes
    # try:
    #     logging.info("🔍 Running keywords analysis...")
    #     keywords.analyze_keywords()
    # except Exception as e:
    #     logging.error(f"❌ Error in keywords analysis: {e}")

    # try:
    #     logging.info("📊 Running sentiment analysis...")
    #     sentiment.analyze_transcripts()
    # except Exception as e:
    #     logging.error(f"❌ Error in sentiment analysis: {e}")

    # try:
    #     logging.info("🌍 Running climate analysis...")
    #     climate.run_climate_analysis()
    # except Exception as e:
    #     logging.error(f"❌ Error in climate analysis: {e}")

    # try:
    #     logging.info("💰 Running financial analysis...")
    #     financials.run_financial_analysis()
    #     financials.run_models()
    # except Exception as e:
    #     logging.error(f"❌ Error in financial analysis: {e}")

    # try:
    #     logging.info("📈 Running graphing and visualization...")
    #     graphing.perform_graphing()
    # except Exception as e:
    #     logging.error(f"❌ Error in graphing: {e}")
