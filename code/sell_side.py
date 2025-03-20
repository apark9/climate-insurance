from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import json
import os

endpoint = "https://sell-side-report-conversion.cognitiveservices.azure.com/"
key = "30Zo5NhnGlGhTaPZmF3ZkeOgWGFXldc3X92bH0aE869MDHfblB1CJQQJ99BCACYeBjFXJ3w3AAAEACOGMADw"

client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))

input_dir = "data/sell_side"
output_dir = "data/sell_side_output"

os.makedirs(output_dir, exist_ok=True)

def process_sellside():
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        
        try:
            with open(input_path, "rb") as f:
                poller = client.begin_analyze_document("prebuilt-read", document=f)
                result = poller.result()
            
            extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])
            
            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(extracted_text)
            
            print(f"Processed: {filename} -> {output_path}")
        
        except HttpResponseError as e:
            print(f"Skipping {filename}: {e.message}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    process_sellside()