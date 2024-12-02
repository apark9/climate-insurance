import logging
import code.sentiment as sentiment  # Import the sentiment module
import code.analysis as analysis   # Import the analysis module

# Configure logging
logging.basicConfig(
    filename="main.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

keywords = ["climate", "catastrophe losses", "weather", "flood", "hurricane", "global warming","earthquake"]

if __name__ == "__main__":
    logging.info("Starting main process...")
    try:
        sentiment.process_sentiment(keywords)
        logging.info("Sentiment analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")

    try:
        analysis.perform_analysis()
        logging.info("Analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")

    logging.info("All processes completed.")
