'''
FILE NOTES:

main.py: runs the sentiment and analysis files


TERMINAL COMMANDS:

bsub -Is -q short_int -n 8 -R "rusage[mem=16G]" python main.py
tail -f main.log

'''


interview_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Insurance_Interviews/"
transcript_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Public_Insurance_Transcripts/"
output_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/output/"

'''
TYPES OF KEYWORDS

WEATHER
keywords = [
    'climate change', 'natural disasters', 'extreme weather', 'hurricane', 'wildfire', 'earthquake', 'flood'
]

FINANCIAL
keywords = [
    'claims cost', 'underwriting losses', 'reinsurance premiums', 'rate increases'
]
'''
keywords = [
    'climate change', 'natural disasters', 'extreme weather', 'hurricane', 'wildfire', 'earthquake', 'flood'
]