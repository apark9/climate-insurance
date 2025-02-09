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

if __name__ == "__main__":
    
    naic.setup_logging()
    naic.analyze_disclosures()

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
