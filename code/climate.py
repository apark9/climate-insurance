import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

states = ["California", "Florida", "Texas"]

data_dir = "data/climate"
output_dir = "output/plots/climate"
analysis_dir = "output/analysis"

states = ["California", "Florida", "Texas"]

def get_quarter_from_month(month):
    if month <= 3:
        return "1Q"
    elif month <= 6:
        return "2Q"
    elif month <= 9:
        return "3Q"
    else:
        return "4Q"

def filter_dataset(use_weighted_time=True):
    file_path = "data/emdat.csv"
    df = pd.read_csv(file_path, dtype=str)

    df.columns = df.columns.str.strip()

    possible_damage_cols = [
        "Total Damage, Adjusted ('000 US$)",
        "Total Damage, Adjusted (000 US$)",
        "Total Damage Adjusted ('000 US$)",
    ]

    damage_col = next((col for col in possible_damage_cols if col in df.columns), None)
    if not damage_col:
        raise KeyError("Could not find the 'Total Damage, Adjusted' column in dataset!")

    for state in states:
        state_df = df[df['Location'].str.contains(state, na=False, case=False)]
        state_df = state_df[['End Year', 'End Month', damage_col]].copy()

        state_df.dropna(subset=['End Year', 'End Month', damage_col], inplace=True)
        state_df['End Year'] = state_df['End Year'].astype(int)
        state_df['End Month'] = state_df['End Month'].astype(int)
        state_df[damage_col] = pd.to_numeric(state_df[damage_col], errors='coerce')

        state_df = state_df[state_df[damage_col] > 0]

        if use_weighted_time:
            weighted_month = (
                state_df.groupby('End Year')
                .apply(lambda x: (x['End Month'] * x[damage_col]).sum() / x[damage_col].sum() if x[damage_col].sum() > 0 else np.nan)
                .reset_index(name='Weighted Month')
            )
            weighted_month.dropna(subset=['Weighted Month'], inplace=True)
            weighted_month['Quarter'] = weighted_month['Weighted Month'].apply(get_quarter_from_month)
            state_df = state_df.groupby(['End Year'], as_index=False)[damage_col].sum()
            state_df = state_df.merge(weighted_month[['End Year', 'Quarter']], on='End Year')
            method_tag = "weighted"
        else:
            state_df['Quarter'] = state_df['End Month'].apply(get_quarter_from_month)
            state_df = state_df.groupby(['End Year', 'Quarter'], as_index=False)[damage_col].sum()
            method_tag = "endperiod"

        if use_weighted_time:
            state_df.columns = ['Year', 'Damages', 'Quarter']
        else:
            state_df.columns = ['Year', 'Quarter', 'Damages']

        output_path = f"{data_dir}/emdat_climate_shocks_{state}_{method_tag}.csv"
        state_df.to_csv(output_path, index=False)

def compute_correlation(state, use_weighted_time=True):
    method_tag = "weighted" if use_weighted_time else "endperiod"
    sentiment_file = f"{data_dir}/sentiment_results_{state}.csv"
    damage_file = f"{data_dir}/emdat_climate_shocks_{state}_{method_tag}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        common_times = set(sentiment_df["Time"]).intersection(set(damage_df["Time"]))
        sentiment_df = sentiment_df[sentiment_df["Time"].isin(common_times)]
        damage_df = damage_df[damage_df["Time"].isin(common_times)]

        sentiment_agg = sentiment_df.groupby("Time", as_index=False).mean(numeric_only=True)
        word_freq_df = sentiment_df[sentiment_df["category"] == "Climate"]
        word_freq_df = word_freq_df.groupby("Time").size().reset_index(name="Keyword Frequency")

        merged_df = sentiment_agg.merge(damage_df, on="Time", how="left")
        keyword_freq_merged = word_freq_df.merge(damage_df, on="Time", how="left")

        correlation_results = {}

        for sentiment_col in ["finbert_negative", "vader_negative", "gpt_negative"]:
            if sentiment_col in merged_df.columns and merged_df["Damages"].notna().sum() > 0:
                pearson_corr, _ = pearsonr(merged_df[sentiment_col].dropna(), merged_df["Damages"].dropna())
                spearman_corr, _ = spearmanr(merged_df[sentiment_col].dropna(), merged_df["Damages"].dropna())
                
                temp_df = merged_df[[sentiment_col, "Damages"]].dropna()
                pandas_corr = temp_df[sentiment_col].corr(temp_df["Damages"])
                
                correlation_results[f"Sentiment ({sentiment_col})"] = {
                    "Pearson": pearson_corr, 
                    "Spearman": spearman_corr,
                    "Pandas Corr": pandas_corr
                }

        if "Keyword Frequency" in keyword_freq_merged.columns and keyword_freq_merged["Damages"].notna().sum() > 0:
            pearson_corr, _ = pearsonr(keyword_freq_merged["Keyword Frequency"].dropna(), keyword_freq_merged["Damages"].dropna())
            spearman_corr, _ = spearmanr(keyword_freq_merged["Keyword Frequency"].dropna(), keyword_freq_merged["Damages"].dropna())

            temp_df = keyword_freq_merged[["Keyword Frequency", "Damages"]].dropna()
            pandas_corr = temp_df["Keyword Frequency"].corr(temp_df["Damages"])
            
            correlation_results["Keyword Frequency"] = {
                "Pearson": pearson_corr,
                "Spearman": spearman_corr,
                "Pandas Corr": pandas_corr
            }

        return correlation_results

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping correlation analysis.")
        return None

def compute_and_save_correlation(use_weighted_time=True):
    method_tag = "weighted" if use_weighted_time else "endperiod"
    correlation_results = {state: compute_correlation(state, use_weighted_time) for state in states}

    output_text = f"Climate Correlation Analysis ({method_tag} method)\n\n"
    for state, results in correlation_results.items():
        if results:
            output_text += f"State: {state}\n"
            for metric, values in results.items():
                output_text += f"  {metric}:\n"
                output_text += f"    Pearson Correlation: {values['Pearson']:.4f}\n"
                output_text += f"    Spearman Correlation: {values['Spearman']:.4f}\n"
                output_text += f"    Pandas Corr: {values['Pandas Corr']:.4f}\n"
            output_text += "\n"

    correlation_file_path = f"{analysis_dir}/climate_correlation_results_{method_tag}.txt"

    with open(correlation_file_path, "w") as file:
        file.write(output_text)

def run_climate_analysis():
    filter_dataset(use_weighted_time=False)
    compute_and_save_correlation(use_weighted_time=False)

    filter_dataset(use_weighted_time=True)
    compute_and_save_correlation(use_weighted_time=True)

def plot_sentiment_vs_damages(state, use_weighted_time=True):
    method_tag = "weighted" if use_weighted_time else "endperiod"
    sentiment_file = f"data/climate/sentiment_results_{state}.csv"
    damage_file = f"data/climate/emdat_climate_shocks_{state}_{method_tag}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        merged_df = sentiment_df.groupby("Time", as_index=False).mean(numeric_only=True).merge(
            damage_df, on="Time", how="left")

        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=merged_df, x="Time", y="finbert_negative", label="FinBERT Sentiment", color="blue")
        sns.lineplot(data=merged_df, x="Time", y="vader_negative", label="VADER Sentiment", color="green")
        sns.lineplot(data=merged_df, x="Time", y="gpt_negative", label="GPT Sentiment", color="red")
        ax2 = ax1.twinx()
        sns.lineplot(data=merged_df, x="Time", y="Damages", label="Damages ($ millions)", color="black", ax=ax2, linestyle="dashed")

        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

        ax1.set_ylabel("Sentiment Score")
        ax2.set_ylabel("Damages ($ millions)")

        plt.xticks(range(0, len(merged_df["Time"]), 4), merged_df["Time"][::4], rotation=45)
        plt.xlabel("Time (Year-Quarter)")
        plt.title(f"Overall Sentiment vs. Damages Over Time - {state} ({method_tag})")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plot_path = f"{output_dir}/sentiment_vs_damages_{state}_{method_tag}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping plot ({method_tag}).")

def plot_climate_sentiment_vs_damages(state, use_weighted_time=True):
    method_tag = "weighted" if use_weighted_time else "endperiod"
    sentiment_file = f"data/climate/sentiment_results_{state}.csv"
    damage_file = f"data/climate/emdat_climate_shocks_{state}_{method_tag}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        climate_df = sentiment_df[sentiment_df["category"] == "Climate"]

        climate_merged_df = climate_df.groupby("Time", as_index=False).mean(numeric_only=True).merge(
            damage_df, on="Time", how="left")

        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=climate_merged_df, x="Time", y="finbert_negative", label="FinBERT Sentiment", color="blue")
        sns.lineplot(data=climate_merged_df, x="Time", y="vader_negative", label="VADER Sentiment", color="green")
        sns.lineplot(data=climate_merged_df, x="Time", y="gpt_negative", label="GPT Sentiment", color="red")
        ax2 = ax1.twinx()
        sns.lineplot(data=climate_merged_df, x="Time", y="Damages", label="Damages ($ millions)", color="black", ax=ax2, linestyle="dashed")

        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

        ax1.set_ylabel("Climate Sentiment Score")
        ax2.set_ylabel("Damages ($ millions)")

        plt.xticks(range(0, len(climate_merged_df["Time"]), 4), climate_merged_df["Time"][::4], rotation=45)
        plt.xlabel("Time (Year-Quarter)")
        plt.title(f"Climate Sentiment vs. Damages Over Time - {state} ({method_tag})")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plot_path = f"{output_dir}/climate_sentiment_vs_damages_{state}_{method_tag}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping plot ({method_tag}).")

def plot_climate_keyword_freq_vs_damages(state, use_weighted_time=True):
    method_tag = "weighted" if use_weighted_time else "endperiod"
    sentiment_file = f"data/climate/sentiment_results_{state}.csv"
    damage_file = f"data/climate/emdat_climate_shocks_{state}_{method_tag}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        keyword_freq = sentiment_df[sentiment_df["category"] == "Climate"]
        keyword_freq = keyword_freq.groupby("Time").size().reset_index(name="Keyword Frequency")

        keyword_freq_merged = keyword_freq.merge(damage_df, on="Time", how="left")

        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=keyword_freq_merged, x="Time", y="Keyword Frequency", label="Climate Keyword Frequency", color="blue")
        ax2 = ax1.twinx()
        sns.lineplot(data=keyword_freq_merged, x="Time", y="Damages", label="Damages ($ millions)", color="black", ax=ax2, linestyle="dashed")

        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

        ax1.set_ylabel("Climate Keyword Frequency")
        ax2.set_ylabel("Damages ($ millions)")

        plt.xticks(range(0, len(keyword_freq_merged["Time"]), 4), keyword_freq_merged["Time"][::4], rotation=45)
        plt.xlabel("Time (Year-Quarter)")
        plt.title(f"Climate Keyword Frequency vs. Damages Over Time - {state} ({method_tag})")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plot_path = f"{output_dir}/climate_keyword_freq_vs_damages_{state}_{method_tag}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping plot ({method_tag}).")

def perform_graphing():
    for state in states:
        for use_weighted_time in [False, True]:
            plot_sentiment_vs_damages(state, use_weighted_time)
            plot_climate_sentiment_vs_damages(state, use_weighted_time)
            plot_climate_keyword_freq_vs_damages(state, use_weighted_time)

