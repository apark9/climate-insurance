import logging
import code.sentiment as sentiment
import code.analysis as analysis

# Configure logging
logging.basicConfig(
    filename="main.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # logging.info("Starting main process...")
    # try:
    #     sentiment.process_sentiment()
    #     logging.info("Sentiment analysis completed successfully.")
    # except Exception as e:
    #     logging.error(f"Error in sentiment analysis: {e}")

    try:
        analysis.perform_analysis()
        logging.info("Analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")

    logging.info("All processes completed.")
