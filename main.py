import logging
import time
import os
import sys
import code.sentiment as sentiment
import code.graphing as graphing
import code.naic as naic
import code.sell_side as sell_side
import code.analysis as analysis
import code.climate as climate
import nltk
if __name__ == "__main__":

    'NAIC'
    # naic.setup_logging()
    # naic.analyze_disclosures()

    'SENTIMENT'
    sentiment.load_finbert() # ALWAYS RUN THIS LINE FOR ANY SENTIMENT

    # batch_number = int(sys.argv[1])
    # sentiment.process_batch(batch_number)
    # sentiment.process_pgr_files()
    # sentiment.merge_batches()
    # sentiment.perform_analysis()
    # sentiment.generate_sentiment_excel()
    # sentiment.compute_quarterly_negative_sentiment()

    'CLIMATE'
    # climate.filter_dataset()
    # climate.run_climate_analysis()
    # climate.perform_graphing()


    'ANALYSIS'
    # analysis.run_analysis()

    'GRAPHING'
    graphing.perform_graphing()