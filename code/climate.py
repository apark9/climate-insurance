import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

# Ensure plots directory exists
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

        logging.info("Disaster data successfully loaded and filtered.")
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

        logging.info("Sentiment data successfully loaded and aggregated.")
        return aggregated_sentiment

    except Exception as e:
        logging.error(f"Error loading sentiment data: {e}")
        raise

def merge_and_analyze(sentiment_data, disaster_data):
    """
    Merge sentiment and disaster data, and perform correlation analysis.
    """
    try:
        logging.info("Merging sentiment and disaster data.")
        merged_data = pd.merge(sentiment_data, disaster_data, on="Year", how="inner")

        # Compute correlation matrix
        correlation_matrix = merged_data.corr()

        # Save data and correlations
        merged_data.to_csv("merged_sentiment_disasters.csv", index=False)
        correlation_matrix.to_csv("sentiment_disaster_correlation.csv", index=True)

        logging.info("Data merged and correlation analysis completed.")
        return merged_data, correlation_matrix

    except Exception as e:
        logging.error(f"Error merging and analyzing data: {e}")
        raise

def plot_trends(merged_data, output_path="plots/sentiment_vs_disasters_trends.png"):
    """
    Plot trends for sentiment and climate disasters.
    """
    try:
        logging.info("Plotting trends.")
        plt.figure(figsize=(12, 6))
        plt.plot(merged_data["Year"], merged_data["compound_sentiment"], label="Sentiment Score", marker="o")
        plt.plot(merged_data["Year"], merged_data["Disaster_Count"], label="Number of Disasters", marker="x")
        plt.plot(merged_data["Year"], merged_data["Total_Damage_Adjusted"] / 1e6, label="Total Damage (Millions)", marker="s")
        plt.title("Sentiment Trends vs Climate Disasters")
        plt.xlabel("Year")
        plt.ylabel("Values (Normalized)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Trends plot saved at {output_path}.")

    except Exception as e:
        logging.error(f"Error plotting trends: {e}")
        raise

def plot_sentiment_vs_disasters(merged_data, output_path="plots/sentiment_vs_disasters_combined.png"):
    """
    Create a multi-axis plot combining sentiment trends, disaster counts, and damage intensity.
    """
    try:
        logging.info("Creating combined multi-axis plot for sentiment vs disasters.")

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot sentiment trends on the primary y-axis
        ax1.plot(
            merged_data["Year"], 
            merged_data["compound_sentiment"], 
            color="blue", 
            marker="o", 
            label="Sentiment Score"
        )
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Sentiment Score", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid()

        # Create secondary y-axis for disaster count and total damage
        ax2 = ax1.twinx()
        ax2.bar(
            merged_data["Year"], 
            merged_data["Disaster_Count"], 
            color="orange", 
            alpha=0.6, 
            label="Disaster Count"
        )
        ax2.plot(
            merged_data["Year"], 
            merged_data["Total_Damage_Adjusted"] / 1e6, 
            color="red", 
            marker="x", 
            linestyle="--", 
            label="Total Damage (Millions)"
        )
        ax2.set_ylabel("Disaster Count / Damage (Millions USD)", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Add a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Add title and tighten layout
        plt.title("Combined Sentiment Trends and Climate Disasters (2000–2024)")
        fig.tight_layout()

        # Save the plot
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Combined multi-axis plot saved at {output_path}.")
    except Exception as e:
        logging.error(f"Error creating combined plot: {e}")
        raise

# Call the function in the pipeline
def run_climate_analysis():
    """
    Run the full climate analysis pipeline and include the combined plot.
    """
    try:
        logging.info("Starting climate analysis pipeline with combined plot.")

        # File paths
        disaster_file_path = 'data/emdat.csv'
        sentiment_file_path = "output/insurance_transcripts.csv"

        # Load data
        disaster_data = load_disaster_data(disaster_file_path)
        sentiment_data = load_sentiment_data(sentiment_file_path)

        # Merge and analyze data
        merged_data, correlation_matrix = merge_and_analyze(sentiment_data, disaster_data)

        # Plot results
        plot_trends(merged_data)
        plot_correlation_heatmap(correlation_matrix)
        plot_sentiment_vs_disasters(merged_data)

        logging.info("Climate analysis pipeline with combined plot completed successfully.")
    except Exception as e:
        logging.error(f"Error in climate analysis pipeline with combined plot: {e}")
        raise

def plot_correlation_heatmap(correlation_matrix, output_path="plots/correlation_heatmap.png"):
    """
    Plot a heatmap of correlations between sentiment and climate disasters.
    """
    try:
        logging.info("Plotting correlation heatmap.")
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Between Sentiment and Climate Disasters")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Correlation heatmap saved at {output_path}.")

    except Exception as e:
        logging.error(f"Error plotting correlation heatmap: {e}")
        raise

if __name__ == "__main__":
    run_climate_analysis()
