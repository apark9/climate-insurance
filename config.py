'''
TERMINAL COMMANDS:

bsub -Is -q short_int -n 8 -R "rusage[mem=16G]" python main.py
tail -f main.log

pip freeze > requirements.txt
'''

interview_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Insurance_Interviews/"
transcript_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Public_Insurance_Transcripts/"
output_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/output/"

keyword_flag = 'financial'

if keyword_flag == 'financial':
    keywords = [
    'catastrophe', 'underwriting losses', 'rate increases', 'premium increases'
    ]
elif keyword_flag == 'climate':
    keywords = [
    'climate change', 'natural disasters', 'extreme weather', 'hurricane', 'wildfire', 'earthquake', 'flood'
]

shifted_flag = 'shifted'