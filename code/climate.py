import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

# Ensure directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_disaster_data(file_path):
    """
    Load and filter disaster data based on relevant keywords.
    """
    try:
        logging.info("Loading disaster data.")
        disaster_data = pd.read_csv(file_path)

        # Filter relevant disasters based on keywords in Disaster Subtype
        relevant_keywords = ["hurricane", "flood", "storm", "tornado"]
        filtered_disasters = disaster_data[
            disaster_data["Disaster Subtype"].str.contains('|'.join(relevant_keywords), case=False, na=False)
        ]

        # Aggregate disaster data by year
        aggregated_disasters = filtered_disasters.groupby("Start Year").agg({
            "Insured Damage, Adjusted ('000 US$)": "sum",
            "Total Damage, Adjusted ('000 US$)": "sum",
            "DisNo.": "count"  # Count number of disasters
        }).reset_index()

        # Rename columns for clarity
        aggregated_disasters.rename(columns={
            "Start Year": "Year",
            "Insured Damage, Adjusted ('000 US$)": "Insured_Damage_Adjusted",
            "Total Damage, Adjusted ('000 US$)": "Total_Damage_Adjusted",
            "DisNo.": "Disaster_Count"
        }, inplace=True)

        # Save to output
        aggregated_disasters.to_csv("output/aggregated_disaster_data.csv", index=False)
        logging.info("Disaster data successfully loaded, filtered, and saved.")
        return aggregated_disasters

    except Exception as e:
        logging.error(f"Error loading disaster data: {e}")
        raise

def load_sentiment_data(file_path):
    """
    Load sentiment data and calculate average sentiment by year.
    """
    try:
        logging.info("Loading sentiment data.")
        sentiment_data = pd.read_csv(file_path)

        # Extract compound sentiment scores
        sentiment_data["compound_sentiment"] = sentiment_data["sentiment"].apply(eval).apply(lambda x: x.get("compound"))

        # Aggregate sentiment data by year
        aggregated_sentiment = sentiment_data.groupby("year").agg({
            "compound_sentiment": "mean"
        }).reset_index().rename(columns={"year": "Year"})

        # Save to output
        aggregated_sentiment.to_csv("output/aggregated_sentiment_data.csv", index=False)
        logging.info("Sentiment data successfully loaded, aggregated, and saved.")
        return aggregated_sentiment

    except Exception as e:
        logging.error(f"Error loading sentiment data: {e}")
        raise

def merge_and_save(sentiment_data, disaster_data):
    """
    Merge sentiment and disaster data and save the merged data.
    """
    try:
        logging.info("Merging sentiment and disaster data.")
        merged_data = pd.merge(sentiment_data, disaster_data, on="Year", how="inner")

        # Save merged data
        merged_data.to_csv("output/merged_sentiment_disasters.csv", index=False)
        logging.info("Merged data saved successfully.")
        return merged_data

    except Exception as e:
        logging.error(f"Error merging and saving data: {e}")
        raise

def run_climate_analysis():
    """
    Run the full climate analysis pipeline.
    """
    try:
        logging.info("Starting climate analysis pipeline.")

        # File paths
        disaster_file_path = "data/emdat.csv"
        sentiment_file_path = "output/insurance_transcripts.csv"

        # Load and process data
        disaster_data = load_disaster_data(disaster_file_path)
        sentiment_data = load_sentiment_data(sentiment_file_path)

        # Merge and save data
        merged_data = merge_and_save(sentiment_data, disaster_data)

        logging.info("Climate analysis pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in climate analysis pipeline: {e}")
        raise

if __name__ == "__main__":
    run_climate_analysis()
