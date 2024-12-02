import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load disaster data
disaster_data = pd.read_excel('/mnt/data/public_emdat.xlsx')

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

# Load sentiment trends
sentiment_data = pd.read_csv("insurance_transcripts.csv")

# Group sentiment data by year and calculate average sentiment
sentiment_data["compound_sentiment"] = sentiment_data["sentiment"].apply(eval).apply(lambda x: x.get("compound"))
aggregated_sentiment = sentiment_data.groupby("year").agg({
    "compound_sentiment": "mean"
}).reset_index().rename(columns={"year": "Year"})

# Merge aggregated sentiment and disaster data
merged_data = pd.merge(aggregated_sentiment, aggregated_disasters, on="Year", how="inner")

# Correlation analysis
correlation_matrix = merged_data.corr()

# Save merged data and correlation matrix
merged_data.to_csv("merged_sentiment_disasters.csv", index=False)
correlation_matrix.to_csv("sentiment_disaster_correlation.csv", index=True)


# Plot trends
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
plt.savefig("sentiment_vs_disasters_trends.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Sentiment and Climate Disasters")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
