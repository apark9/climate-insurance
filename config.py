'''
TERMINAL COMMANDS:

general use:
    bsub -q short -n 8 -R "rusage[mem=32G]" -u averypark@college.harvard.edu -N python main.py
    tail -f main.log
    source myenv/bin/activate

bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_0_200.log -N -u averypark@college.harvard.edu python main.py --start_index 0 --end_index 200
bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_200_400.log -N -u averypark@college.harvard.edu python main.py --start_index 200 --end_index 400
bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_400_600.log -N -u averypark@college.harvard.edu python main.py --start_index 400 --end_index 600
bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_600_800.log -N -u averypark@college.harvard.edu python main.py --start_index 600 --end_index 800
bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_800_1000.log -N -u averypark@college.harvard.edu python main.py --start_index 800 --end_index 1000
bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_1000_1200.log -N -u averypark@college.harvard.edu python main.py --start_index 1000 --end_index 1200
bsub -q short -n 8 -R "rusage[mem=32G]" -e logs/job_error_1200_1353.log -N -u averypark@college.harvard.edu python main.py --start_index 1200 --end_index 1353
    
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