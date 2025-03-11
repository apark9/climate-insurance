import os
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# Define input and output folder
input_folder = "analysis/regressions"
output_folder = "analysis/regressions"

# Define models and sentiment variables
models = ["FinBERT", "VADER", "GPT"]
sentiment_vars = ["Sentiment_All", "Sentiment_Climate"]

# Store file paths for each model
input_files = {model: os.path.join(input_folder, f"{model}.xlsx") for model in models}
output_files = {model: os.path.join(output_folder, f"regression_results_{model}.txt") for model in models}

# Function to run fixed effects regression
def run_fixed_effects_regression(df, dependent_var):
    model = PanelOLS.from_formula(
        f"{dependent_var} ~ P_Written + P_Earned + Loss_Ratio + Loss_Containment_Ratio + Market_Share + Stock_Price + Total_Damage_Adj + EntityEffects",
        data=df
    )
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results

# Run regressions for each model
for model in models:
    file_path = input_files[model]
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}, skipping {model}...")
        continue
    
    # Load the data
    df = pd.read_excel(file_path, sheet_name="Sheet 1")
    df.set_index(["Company", "Year"], inplace=True)
    
    # Standardize independent variables
    scaler = StandardScaler()
    columns_to_standardize = ["P_Written", "P_Earned", "Loss_Ratio", "Loss_Containment_Ratio", "Market_Share", "Stock_Price", "Total_Damage_Adj"]
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    
    regression_results = {}
    for sentiment in sentiment_vars:
        sentiment_col = f"{model}_{sentiment}"  # Adjust column names based on the model
        if sentiment_col in df.columns:
            regression_results[sentiment] = run_fixed_effects_regression(df, sentiment_col)
            print(f"Results for {model} - {sentiment}:")
            print(regression_results[sentiment].summary)
        else:
            print(f"⚠️ Column {sentiment_col} not found in dataset, skipping...")
    
    # Save results for each model separately
    with open(output_files[model], "w") as f:
        for sentiment, result in regression_results.items():
            f.write(f"Results for {model} - {sentiment}:\n")
            f.write(str(result.summary))
            f.write("\n\n")
    print(f"✅ Regression results saved to {output_files[model]}")