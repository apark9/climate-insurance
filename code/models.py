import pandas as pd
from datetime import datetime
from config import key_line_items_afg, key_line_items_afl, key_line_items_all, key_line_items_pgr, key_line_items_trv
import os

def format_row_4_as_dates(header):
    try:
        # Extract row 4 (assuming 0-indexing, this would be row index 3)
        row_4 = header.iloc[3, :].astype(str)

        # Strip time and convert to datetime objects, handle non-date entries gracefully
        def parse_date(value):
            try:
                return datetime.strptime(value.split()[0], '%m/%d/%Y').strftime('%Y-%m-%d')
            except ValueError:
                return value  # Return the original value if it is not a valid date

        formatted_dates = row_4.apply(parse_date)

        # Replace row 4 with formatted dates by creating a copy
        header = header.copy()
        header.iloc[3, :] = formatted_dates
    except Exception as e:
        print(f"ERROR - Error formatting row 4: {e}")
    return header

def filter_and_export(file_path, key_items, output_path):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path, sheet_name='Model', header=None, engine='openpyxl')

        # Extract the header (e.g., row with dates or other details)
        header = df.iloc[:4, :].copy()  # Assume the first 4 rows contain the header
        header = format_row_4_as_dates(header)  # Format row 4 dates
        data = df.iloc[4:, :].copy()    # Remaining rows contain the data

        # Filter rows where the first column matches the key items
        filtered_data = data[data.iloc[:, 0].isin(key_items)]

        # Reattach the header
        filtered_df = pd.concat([header, filtered_data], ignore_index=True)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export to a new Excel file
        filtered_df.to_excel(output_path, index=False, header=False, engine='openpyxl')
    except Exception as e:
        print(f"ERROR - Error in financial analysis: {e}")

def run_models():
    # File paths for the input Excel files with full descriptive names
    file_paths = {
        "AFG": "data/Financial_Models/American Financial Group AFG US.xlsx",
        "TRV": "data/Financial_Models/The Travelers Companies TRV US.xlsx",
        "AFL": "data/Financial_Models/Aflac AFL US.xlsx",
        "PGR": "data/Financial_Models/The Progressive Corporation PGR US.xlsx",
        "ALL": "data/Financial_Models/The Allstate Corporation ALL US.xlsx"
    }

    # Corresponding output paths in the "output" folder
    output_paths = {
        "AFG": "output/American_Financial_Group_AFG_US.xlsx",
        "TRV": "output/The_Travelers_Companies_TRV_US.xlsx",
        "AFL": "output/Aflac_AFL_US.xlsx",
        "PGR": "output/The_Progressive_Corporation_PGR_US.xlsx",
        "ALL": "output/The_Allstate_Corporation_ALL_US.xlsx"
    }

    filter_and_export(file_paths["AFG"], key_line_items_afg, output_paths["AFG"])
    filter_and_export(file_paths["TRV"], key_line_items_trv, output_paths["TRV"])
    filter_and_export(file_paths["AFL"], key_line_items_afl, output_paths["AFL"])
    filter_and_export(file_paths["PGR"], key_line_items_pgr, output_paths["PGR"])
    filter_and_export(file_paths["ALL"], key_line_items_all, output_paths["ALL"])

if __name__ == "__main__":
    run_models()
