import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from config import keyword_flag

os.makedirs("output", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_disaster_data(file_path):
    try:
        logging.info("Loading disaster data.")
        disaster_data = pd.read_csv(file_path)

        relevant_keywords = ["hurricane", "flood", "storm", "tornado"]
        filtered_disasters = disaster_data[
            disaster_data["Disaster Subtype"].str.contains('|'.join(relevant_keywords), case=False, na=False)
        ]

        aggregated_disasters = filtered_disasters.groupby("Start Year").agg({
            "Insured Damage, Adjusted ('000 US$)": "sum",
            "Total Damage, Adjusted ('000 US$)": "sum",
            "DisNo.": "count"
        }).reset_index()

        aggregated_disasters.rename(columns={
            "Start Year": "Year",
            "Insured Damage, Adjusted ('000 US$)": "Insured_Damage_Adjusted",
            "Total Damage, Adjusted ('000 US$)": "Total_Damage_Adjusted",
            "DisNo.": "Disaster_Count"
        }, inplace=True)

        aggregated_disasters.to_csv(f"output/aggregated_disaster_data_{keyword_flag}.csv", index=False)
        logging.info("Disaster data successfully loaded, filtered, and saved.")

    except Exception as e:
        logging.error(f"Error loading disaster data: {e}")
        raise

def run_climate_analysis():
    try:
        logging.info("Starting climate analysis pipeline.")
        disaster_file_path = "data/emdat.csv"
        load_disaster_data(disaster_file_path)
        
        logging.info("Climate analysis pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in climate analysis pipeline: {e}")
        raise

if __name__ == "__main__":
    run_climate_analysis()
