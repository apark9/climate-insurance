import logging
import code.sentiment as sentiment
import code.graphing as graphing
import code.climate as climate

# Configure logging
logging.basicConfig(
    filename="main.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.info("Starting main process...")
    # try:
    #     sentiment.process_sentiment()
    #     logging.info("Sentiment analysis completed successfully.")
    # except Exception as e:
    #     logging.error(f"Error in sentiment analysis: {e}")

    # try:
    #     graphing.perform_graphing()
    #     logging.info("Graphing completed successfully.")
    # except Exception as e:
    #     logging.error(f"Error in graphing: {e}")

    try:
        climate.run_climate_analysis()
        logging.info("Climate analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in graphing: {e}")

    logging.info("All processes completed.")
