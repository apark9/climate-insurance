'''
TERMINAL COMMANDS:

general use:
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
        'catastrophe', 'underwriting losses', 'rate increases', 'premium increases',
        'reinsurance', 'claims reserves', 'risk modeling', 'loss ratios',
        'carbon pricing', 'sustainability initiatives', 'ESG investing'
    ]
elif keyword_flag == 'climate':
    keywords = [
        'climate change', 'natural disasters', 'extreme weather', 'hurricane',
        'wildfire', 'earthquake', 'flood', 'coastal flooding', 'sea level rise',
        'drought', 'adaptation strategies', 'climate resilience', 'GHG emissions',
        'green energy transition', 'net zero commitments'
    ]


company_keywords = {
    "Progressive": ['"Progressive"'],
    "AllState": ['"AllState"'],
    "Aflac": ['"Aflac"'],
    "American Financial Group": ['"American Financial Group"'],
    "Travelers": ['"Travelers"']
}

general_keywords = [
    '"insurers exiting states"',
    '"climate risk insurance adaptation"',
    '"reinsurance challenges"',
    '"climate risk affecting insurers"',
    '"insurance climate exit"'
]