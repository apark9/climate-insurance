import logging
import time
import os
import sys
import code.sentiment as sentiment
import code.graphing as graphing
import nltk

# import code.climate as climate
# import code.financials as financials
# import code.keywords as keywords
# import code.naic as naic

if __name__ == "__main__":

    'NAIC'
    # naic.setup_logging()
    # naic.analyze_disclosures()

    # batch_number = int(sys.argv[1])

    'SENTIMENT'
    # sentiment.process_batch(batch_number)
    # sentiment.merge_batches()
    # sentiment.perform_analysis()

    'GRAPHING'
    graphing.perform_graphing()
