import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

input_folder = "analysis/regressions"
output_folder = "analysis/regressions"

models = ["FinBERT", "VADER", "GPT"]
sentiment_vars = ["Sentiment_All", "Sentiment_Climate", "Sentiment_Risk", "Sentiment_Financial"]

input_files = {model: os.path.join(input_folder, f"{model}.xlsx") for model in models}
output_files = {model: os.path.join(output_folder, f"regression_results_{model}.txt") for model in models}

def run_fixed_effects_regression(df, dependent_var, fixed_effects=True, time_fixed_effects=True):
    df.columns = df.columns.str.strip().str.lower()
    dependent_var = dependent_var.lower()

    if dependent_var not in df.columns:
        raise ValueError(f"{dependent_var} not found in dataset columns: {df.columns.tolist()}")

    if "year" not in df.columns:
        time_fixed_effects = False  

    if time_fixed_effects and "year" in df.columns:
        df = pd.get_dummies(df, columns=["year"], drop_first=True)

    year_dummies = " + ".join([col for col in df.columns if col.startswith("year_")])
    formula = f"{dependent_var} ~ p_written + p_earned + loss_ratio + loss_containment_ratio + market_share + stock_price + total_damage_adj"

    if fixed_effects:
        formula += " + EntityEffects"

    if time_fixed_effects and year_dummies:
        formula += f" + {year_dummies}"

    model = PanelOLS.from_formula(formula, data=df)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

def run_fixed_time_effects_regression(df, dependent_var, fixed_effects=True, time_fixed_effects=True):
    df.columns = df.columns.str.strip().str.lower()
    dependent_var = dependent_var.lower()

    if dependent_var not in df.columns:
        raise ValueError(f"{dependent_var} not found in dataset columns: {df.columns.tolist()}")

    if time_fixed_effects and "year" not in df.columns:
        time_fixed_effects = False  

    if "company" not in df.index.names:
        raise ValueError("'Company' must be set as an index for panel data.")

    independent_vars = ["p_written", "p_earned", "loss_ratio", "loss_containment_ratio", "market_share", "stock_price", "total_damage_adj"]
    formula = f"{dependent_var} ~ " + " + ".join(independent_vars)

    model = PanelOLS.from_formula(
        formula,
        data=df,
        entity_effects=fixed_effects,
        time_effects=time_fixed_effects
    )

    results = model.fit(cov_type='clustered', cluster_entity=True)

    return results

def plot_coefficients(model, sentiment, results_fe, results_no_fe):
    params_fe = results_fe.params.drop("Intercept", errors="ignore")
    params_no_fe = results_no_fe.params.drop("Intercept", errors="ignore")
    
    df_plot = pd.DataFrame({
        "Fixed Effects": params_fe,
        "No Fixed Effects": params_no_fe
    })
    
    df_plot.plot(kind='bar', figsize=(10, 6), edgecolor='black')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title(f"Coefficient Comparison for {model} - {sentiment}")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Variables")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f"{model}_{sentiment}_coefficients_comparison.png")
    plt.savefig(plot_path)
    plt.close()

def run_analysis():
    for model in models:
        file_path = input_files[model]
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path, sheet_name="Sheet 1")
        df.set_index(["Company", "Year"], inplace=True)

        if "Year" in df.index.names:
            df["year"] = df.index.get_level_values("Year")
            if df["year"].isnull().any():
                most_common_year = df["year"].mode()[0]
                df["year"] = df["year"].fillna(most_common_year)
            df["year"] = df["year"].astype(int)
        else:
            continue

        scaler = StandardScaler()
        columns_to_standardize = ["P_Written", "P_Earned", "Loss_Ratio", "Loss_Containment_Ratio", "Market_Share", "Stock_Price", "Total_Damage_Adj"]
        df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

        df.columns = df.columns.str.strip().str.lower()
        sentiment_vars_lower = [col.lower() for col in sentiment_vars]

        regression_results = {}
        for sentiment in sentiment_vars_lower:
            if sentiment in df.columns:
                regression_results[sentiment] = {}

                results_entity_fe = run_fixed_effects_regression(df, sentiment, fixed_effects=True, time_fixed_effects=False)
                results_time_fe = run_fixed_effects_regression(df, sentiment, fixed_effects=True, time_fixed_effects=True)

                regression_results[sentiment]['Entity Fixed Effects Only'] = results_entity_fe
                regression_results[sentiment]['Entity + Time Fixed Effects'] = results_time_fe
                
                plot_coefficients(model, sentiment, results_entity_fe, results_time_fe)

        with open(output_files[model], "w") as f:
            for sentiment, results in regression_results.items():
                for fe_type, result in results.items():
                    f.write(f"Results for {model} - {sentiment} ({fe_type}):\n")
                    f.write(str(result.summary))
                    f.write("\n\n")

if __name__ == "__main__":
    run_analysis()