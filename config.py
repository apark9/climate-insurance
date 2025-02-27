'''
TERMINAL COMMANDS:

general use:
    bsub -q long -n 8 -R "rusage[mem=48G]" -e logs/job_error_%J.log -u averypark@college.harvard.edu -N python main.py
    source myenv/bin/activate

    for i in {1..21}; do   bsub -q long -n 8 -R "rusage[mem=48G]"        -e logs/job_error_%J_batch_$i.log        -o logs/job_output_%J_batch_$i.log        -u averypark@college.harvard.edu -N        python main.py $i; done
    
    for i in {1..21}; do   bsub -q long -n 8 -R "rusage[mem=48G]"        python main.py $i; done
pip freeze > requirements.txt
'''

interview_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Insurance_Interviews/"
transcript_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Public_Insurance_Transcripts/"
output_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/output/"

keywords_financial = [
    'catastrophe', 'underwriting losses', 'rate increases', 'premium increases',
    'policy cancellations', 'claims reserves', 'risk modeling', 'loss ratios',
    'insurance payouts', 'capital adequacy', 'financial risk', 'market exposure',
    'insurance insolvency', 'portfolio risk', 'actuarial adjustments',
    'carbon pricing', 'sustainability initiatives', 'ESG investing',
    'reinsurance', 'insurance risk transfer', 'capital markets'
]

keywords_climate = [
    'climate change', 'natural disasters', 'extreme weather', 'hurricane',
    'wildfire', 'earthquake', 'flood', 'coastal flooding', 'sea level rise',
    'drought', 'climate adaptation', 'climate resilience', 'GHG emissions',
    'carbon emissions', 'greenhouse gases', 'carbon footprint',
    'green energy transition', 'net zero commitments', 'carbon offsets',
    'sustainability risks', 'climate mitigation', 'disaster recovery'
]

keywords_risk = [
    'climate risk', 'physical risk', 'transition risk', 'financial exposure',
    'economic losses', 'insured losses', 'uninsured losses', 'capital flight',
    'infrastructure damage', 'business interruption', 'supply chain disruption',
    'market volatility', 'risk assessment', 'stress testing', 'scenario analysis'
]

