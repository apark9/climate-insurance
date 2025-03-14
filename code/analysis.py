import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# Define input and output folder
input_folder = "analysis/regressions"
output_folder = "analysis/regressions"

# Define models and sentiment variables
models = ["FinBERT", "VADER", "GPT"]
sentiment_vars = ["Sentiment_All", "Sentiment_Climate", "Sentiment_Risk", "Sentiment_Financial"]

# Store file paths for each model
input_files = {model: os.path.join(input_folder, f"{model}.xlsx") for model in models}
output_files = {model: os.path.join(output_folder, f"regression_results_{model}.txt") for model in models}

# Function to run fixed effects regression
def run_fixed_effects_regression(df, dependent_var, fixed_effects=True):
    formula = f"{dependent_var} ~ P_Written + P_Earned + Loss_Ratio + Loss_Containment_Ratio + Market_Share + Stock_Price + Total_Damage_Adj"
    if fixed_effects:
        formula += " + EntityEffects"
    
    model = PanelOLS.from_formula(formula, data=df)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results

# Function to extract coefficients from regression results and plot them
def plot_coefficients(model, sentiment, results_fe, results_no_fe):
    params_fe = results_fe.params.drop("Intercept", errors="ignore")  # Remove intercept if present
    params_no_fe = results_no_fe.params.drop("Intercept", errors="ignore")
    
    df_plot = pd.DataFrame({
        "Fixed Effects": params_fe,
        "No Fixed Effects": params_no_fe
    })
    
    df_plot.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'], edgecolor='black')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title(f"Coefficient Comparison for {model} - {sentiment}")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Variables")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_folder, f"{model}_{sentiment}_coefficients_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Coefficient comparison plot saved to {plot_path}")

# Run analysis for each model
def run_analysis():
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
            sentiment_col = f"{sentiment}"  # Adjust column names based on the model
            if sentiment_col in df.columns:
                regression_results[sentiment] = {}
                
                # Run regressions with and without fixed effects
                results_fe = run_fixed_effects_regression(df, sentiment_col, fixed_effects=True)
                results_no_fe = run_fixed_effects_regression(df, sentiment_col, fixed_effects=False)
                
                regression_results[sentiment]['Fixed Effects'] = results_fe
                regression_results[sentiment]['No Fixed Effects'] = results_no_fe
                
                print(f"Results for {model} - {sentiment} (Fixed Effects):")
                print(results_fe.summary)
                print(f"Results for {model} - {sentiment} (No Fixed Effects):")
                print(results_no_fe.summary)
                
                # Plot coefficient comparisons
                plot_coefficients(model, sentiment, results_fe, results_no_fe)
            else:
                print(f"⚠️ Column {sentiment_col} not found in dataset, skipping...")
        
        # Save results for each model separately
        with open(output_files[model], "w") as f:
            for sentiment, results in regression_results.items():
                for fe_type, result in results.items():
                    f.write(f"Results for {model} - {sentiment} ({fe_type}):\n")
                    f.write(str(result.summary))
                    f.write("\n\n")
        print(f"✅ Regression results saved to {output_files[model]}")

if __name__ == "__main__":
    run_analysis()
