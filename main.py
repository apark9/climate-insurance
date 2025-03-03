import logging
import time
import os
import sys
import code.sentiment as sentiment
import code.graphing as graphing
import code.naic as naic
import nltk

# import code.climate as climate
# import code.financials as financials
# import code.keywords as keywords

if __name__ == "__main__":

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/main.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
    )

    'NAIC'
    # naic.setup_logging()
    # naic.analyze_disclosures()

    'SENTIMENT'
    # try:
    #     batch_index = int(sys.argv[1])
    # except ValueError:
    #     logging.error("Batch index must be an integer.")
    #     sys.exit(1)
    # naic.analyze_single_batch(batch_index)

    # batch_number = int(sys.argv[1])
    # sentiment.process_batch(batch_number)
    # sentiment.merge_batches()
    # sentiment.perform_analysis()
    sentiment.generate_sentiment_excel()

    'GRAPHING'
    # graphing.perform_graphing()
